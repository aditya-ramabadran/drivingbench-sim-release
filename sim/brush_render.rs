use anyhow::Result;
use brush_render::{
    MainBackend, SplatForward, burn_init_setup,
    camera::{Camera, focal_to_fov},
    gaussian_splats::Splats,
    shaders::project_visible::SH_C0,
};
use brush_serde::load_splat_from_ply;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use image::{ExtendedColorType, codecs::jpeg::JpegEncoder};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufRead, Write},
    path::Path,
};

#[derive(Deserialize)]
struct View {
    name: String,
    transform_matrix: [[f32; 4]; 4],
}

#[derive(Clone, Deserialize)]
struct Intrinsics {
    w: u32,
    h: u32,
    fl_x: f64,
    fl_y: f64,
    cx: f64,
    cy: f64,
}

#[derive(Deserialize)]
struct Views {
    #[serde(flatten)]
    camera: Intrinsics,
    frames: Vec<View>,
    #[serde(default)]
    depth: bool,
}

#[derive(Deserialize)]
struct Request {
    #[serde(rename = "id")]
    _id: Value,
    transform_matrix: [[f32; 4]; 4],
    camera: Intrinsics,
    output: String,
}

fn make_camera(matrix: [[f32; 4]; 4], data: &Intrinsics, opencv: bool) -> Result<Camera> {
    anyhow::ensure!(
        data.w > 0 && data.h > 0,
        "Image dimensions must be positive"
    );
    anyhow::ensure!(
        [data.fl_x, data.fl_y, data.cx, data.cy]
            .iter()
            .all(|v| v.is_finite())
            && data.fl_x > 0.0
            && data.fl_y > 0.0,
        "Intrinsics must be finite and focal lengths positive"
    );
    let values: Vec<f32> = matrix.into_iter().flatten().collect();
    anyhow::ensure!(
        values.iter().all(|v| v.is_finite()),
        "Transform must be finite"
    );
    let mut transform = glam::Mat4::from_cols_slice(&values).transpose();
    let rotation = glam::Mat3::from_mat4(transform);
    let error = rotation.transpose() * rotation - glam::Mat3::IDENTITY;
    anyhow::ensure!(
        error.to_cols_array().iter().all(|v| v.abs() < 1e-4)
            && (rotation.determinant() - 1.0).abs() < 1e-4
            && matrix[3]
                .iter()
                .zip([0.0, 0.0, 0.0, 1.0])
                .all(|(a, b)| (a - b).abs() < 1e-5),
        "Transform must be rigid and right-handed"
    );
    if !opencv {
        transform.y_axis *= -1.0;
        transform.z_axis *= -1.0;
    }
    let (_, rotation, translation) = transform.to_scale_rotation_translation();
    Ok(Camera::new(
        translation,
        rotation,
        focal_to_fov(data.fl_x, data.w),
        focal_to_fov(data.fl_y, data.h),
        glam::vec2(
            (data.cx / data.w as f64) as f32,
            (data.cy / data.h as f64) as f32,
        ),
    ))
}

fn npy_header(file: &mut File, description: &str, shape: &str) -> Result<()> {
    let header =
        format!("{{'descr': '{description}', 'fortran_order': False, 'shape': {shape}, }}");
    let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
    let header = format!("{}{}\n", header, " ".repeat(padding));
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    Ok(())
}

fn save_rgb(path: &Path, size: glam::UVec2, data: &[u8]) -> Result<()> {
    let mut file = File::create(path)?;
    npy_header(&mut file, "|u1", &format!("({}, {}, 3)", size.y, size.x))?;
    file.write_all(data)?;
    Ok(())
}

fn respond(response: Value) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "DRIVINGBENCH {response}")?;
    stdout.flush()?;
    Ok(())
}

async fn serve(splats: &Splats<MainBackend>, positions: &[f32]) -> Result<()> {
    respond(json!({"ready": true, "protocol": 1, "camera_convention": "OpenCV camera-to-world"}))?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let value = serde_json::from_str::<Value>(&line);
        let id = value
            .as_ref()
            .ok()
            .and_then(|v| v.get("id"))
            .cloned()
            .unwrap_or(Value::Null);
        let started = std::time::Instant::now();
        let result: Result<(f64, f64)> = async {
            let request: Request = serde_json::from_value(value?)?;
            let camera = make_camera(request.transform_matrix, &request.camera, true)?;
            let size = glam::uvec2(request.camera.w, request.camera.h);
            anyhow::ensure!(!request.output.is_empty(), "Output directory is required");
            let out = Path::new(&request.output);
            std::fs::create_dir_all(out)?;
            let (rendered, _) = splats.render(&camera, size, glam::Vec3::ZERO, None);
            let rgba = rendered.into_data_async().await.bytes;
            let rgb: Vec<u8> = rgba
                .chunks_exact(4)
                .flat_map(|p| p[..3].iter().copied())
                .collect();
            let (depth, alpha) = expected_depth(splats, positions, &camera, size).await?;
            let rendered_s = started.elapsed().as_secs_f64();
            save_rgb(&out.join("rgb.npy"), size, &rgb)?;
            save_npy(&out.join("depth.npy"), size, &depth)?;
            save_npy(&out.join("alpha.npy"), size, &alpha)?;
            Ok((rendered_s, started.elapsed().as_secs_f64() - rendered_s))
        }
        .await;
        match result {
            Ok((render_s, write_s)) => respond(
                json!({"id": id, "ok": true, "elapsed_s": started.elapsed().as_secs_f64(), "render_s": render_s, "write_s": write_s}),
            )?,
            Err(error) => respond(
                json!({"id": id, "ok": false, "error": error.to_string(), "elapsed_s": started.elapsed().as_secs_f64()}),
            )?,
        }
    }
    Ok(())
}

fn save_npy(path: &Path, size: glam::UVec2, data: &[f32]) -> Result<()> {
    let mut file = File::create(path)?;
    npy_header(&mut file, "<f4", &format!("({}, {})", size.y, size.x))?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    file.write_all(&bytes)?;
    Ok(())
}

async fn expected_depth(
    splats: &Splats<MainBackend>,
    positions: &[f32],
    camera: &Camera,
    size: glam::UVec2,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let world_to_camera = camera.world_to_local();
    let coefficients: Vec<f32> = positions
        .chunks_exact(3)
        .flat_map(|p| {
            let z = world_to_camera
                .transform_point3(glam::Vec3::from_slice(p))
                .z;
            [(z - 0.5) / SH_C0; 3]
        })
        .collect();
    let dc: Tensor<MainBackend, 3> = Tensor::from_data(
        TensorData::new(coefficients, [positions.len() / 3, 1, 3]),
        &splats.means.val().device(),
    );
    // The float raster path accumulates unclipped metric z with the original visibility weights.
    let (raw, _) = MainBackend::render_splats(
        camera,
        size,
        splats.means.val().into_primitive().tensor(),
        splats.log_scales.val().into_primitive().tensor(),
        splats.rotation.val().into_primitive().tensor(),
        dc.into_primitive().tensor(),
        splats.raw_opacity.val().into_primitive().tensor(),
        glam::Vec3::ZERO,
        true,
    );
    let pixels: Vec<f32> = Tensor::<MainBackend, 3>::from_primitive(TensorPrimitive::Float(raw))
        .into_data_async()
        .await
        .to_vec()
        .map_err(|e| anyhow::anyhow!("Depth readback: {e:?}"))?;
    let depth = pixels
        .chunks_exact(4)
        .map(|p| {
            if p[3] >= 0.5 && p[0].is_finite() {
                p[0] / p[3]
            } else {
                f32::NAN
            }
        })
        .collect();
    let alpha = pixels.chunks_exact(4).map(|p| p[3]).collect();
    Ok((depth, alpha))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        (args.len() == 4 && args[2] != "--serve") || (args.len() == 3 && args[2] == "--serve"),
        "Usage: drivingbench-brush-render SPLATS.ply CAMERAS.json OUT | SPLATS.ply --serve"
    );
    let streaming = args[2] == "--serve";
    let views: Option<Views> = if streaming {
        None
    } else {
        Some(serde_json::from_reader(File::open(&args[2])?)?)
    };
    let device = burn_init_setup().await;
    let splats = load_splat_from_ply(tokio::fs::File::open(&args[1]).await?, None, device)
        .await?
        .splats;
    let positions: Vec<f32> = if streaming || views.as_ref().is_some_and(|v| v.depth) {
        splats
            .means
            .val()
            .into_data_async()
            .await
            .to_vec()
            .map_err(|e| anyhow::anyhow!("Splat positions: {e:?}"))?
    } else {
        Vec::new()
    };
    if streaming {
        return serve(&splats, &positions).await;
    }
    let views = views.expect("batch views");
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out)?;
    for view in views.frames {
        let started = std::time::Instant::now();
        let camera = make_camera(view.transform_matrix, &views.camera, false)?;
        let (rendered, _) = splats.render(
            &camera,
            glam::uvec2(views.camera.w, views.camera.h),
            glam::Vec3::ZERO,
            None,
        );
        // Brush's inference tensor contains packed RGBA8, not numeric floats.
        let rgba = rendered.into_data_async().await.bytes;
        let rgb: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|p| p[..3].iter().copied())
            .collect();
        let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
        JpegEncoder::new_with_quality(File::create(out.join(format!("{}.jpg", view.name)))?, 92)
            .encode(
                &rgb,
                views.camera.w,
                views.camera.h,
                ExtendedColorType::Rgb8,
            )?;
        image::save_buffer(
            out.join(format!("{}_color.png", view.name)),
            &rgb,
            views.camera.w,
            views.camera.h,
            image::ColorType::Rgb8,
        )?;
        image::save_buffer(
            out.join(format!("{}_alpha.png", view.name)),
            &alpha,
            views.camera.w,
            views.camera.h,
            image::ColorType::L8,
        )?;
        if views.depth {
            let size = glam::uvec2(views.camera.w, views.camera.h);
            let (depth, alpha) = expected_depth(&splats, &positions, &camera, size).await?;
            save_npy(&out.join(format!("{}_depth.npy", view.name)), size, &depth)?;
            save_npy(
                &out.join(format!("{}_depth_alpha.npy", view.name)),
                size,
                &alpha,
            )?;
        }
        println!("{} {:.6}", view.name, started.elapsed().as_secs_f64());
    }
    Ok(())
}
