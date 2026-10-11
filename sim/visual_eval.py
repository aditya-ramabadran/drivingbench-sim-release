"""Full-resolution, paired image scores and coverage for frozen held-out cameras."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def image(path):
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.float32) / 255


def coverage(query, training):
    """Distance and view changes relative to the closest training camera on the ground."""
    xy = np.array([[r["x"], r["y"]] for r in training])
    distance = np.linalg.norm(xy - [query["x"], query["y"]], axis=1)
    k = int(np.argmin(distance))
    reference = training[k]
    heading = reference["heading"]
    delta = np.array([query["x"], query["y"]]) - xy[k]
    angle = (query["heading"] - heading + np.pi) % (2 * np.pi) - np.pi
    return {
        "nearest_train": reference["name"],
        "distance_m": float(distance[k]),
        "lateral_m": float(abs(delta @ [-np.sin(heading), np.cos(heading)])),
        "heading_deg": float(abs(np.degrees(angle))),
    }


def pixel_scores(reference, rendered):
    from skimage.metrics import structural_similarity

    if reference.shape != rendered.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Image metrics require matching RGB dimensions; resizing is not allowed")
    mse = float(np.mean((reference.astype(np.float64) - rendered) ** 2))
    return {
        "psnr_db": float(-10 * np.log10(mse)) if mse else None,
        "identical": mse == 0,
        "ssim": float(
            structural_similarity(
                reference,
                rendered,
                channel_axis=2,
                data_range=1,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ),
    }


def summarize(rows, names):
    summary = {}
    for name in names:
        scores = [r["scores"][name] for r in rows if "error" not in r["scores"][name]]
        exact = sum(s["identical"] for s in scores)
        summary[name] = {"scored": len(scores), "requested": len(rows), "identical": exact}
        for key in ("psnr_db", "ssim", "lpips_alex"):
            finite = [s[key] for s in scores if s[key] is not None]
            summary[name][key] = float(np.mean(finite)) if finite else None
        if exact:
            summary[name]["psnr_db"] = None
            summary[name]["psnr_infinite"] = True
    return summary


def evaluation_training(manifest, queries):
    train = [r for r in manifest if r["split"] == "train"]
    held_out = {r["name"]: r for r in manifest if r["split"] == "test"}
    train_paths = {Path(r["file_path"]).resolve() for r in train}
    if not train or not queries or len({q["name"] for q in queries}) != len(queries):
        raise ValueError("Evaluation needs training and unique held-out cameras")
    for query in queries:
        reference = held_out.get(query["name"])
        if (
            reference is None
            or query["split"] != "test"
            or "perturbation" in query
            or Path(query["file_path"]).resolve() in train_paths
            or Path(query["file_path"]).resolve() != Path(reference["file_path"]).resolve()
        ):
            raise ValueError(
                "Evaluation requires observed, disjoint held-out cameras; synthetic offsets have no truth"
            )
    return train


def figures(manifest, queries, renders, out):
    """Chronologically spaced comparisons and measured held-out camera coverage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    chosen = [queries[i] for i in np.linspace(0, len(queries) - 1, 3, dtype=int)]
    fig, axes = plt.subplots(3, len(renders) + 1, figsize=(4 * (len(renders) + 1), 7.5))
    for i, query in enumerate(chosen):
        paths = {"Real": Path(query["file_path"])} | {
            k: p / query["name"] for k, p in renders.items()
        }
        for ax, (name, path) in zip(axes[i], paths.items()):
            ax.imshow(image(path))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name if i == 0 else "", fontsize=12)
        axes[i, 0].set_ylabel(f"t={query['t']:.1f} s", fontsize=11)
    fig.tight_layout(pad=0.6)
    fig.savefig(out / "visual-comparison.jpg", dpi=110)
    plt.close(fig)

    train = [r for r in manifest if r["split"] == "train"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, label, color in (
        (train, "Training cameras", "#777777"),
        (queries, "Held-out cameras", "#da6a1e"),
    ):
        ax.scatter([r["x"] for r in rows], [r["y"] for r in rows], s=10, c=color, label=label)
    ax.set(
        xlabel="East (m)",
        ylabel="North (m)",
        title="Site-1 pilot: frozen-map camera poses",
        aspect="equal",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "visual-coverage.png", dpi=140)
    plt.close(fig)


def novel_figure(queries, renders, out):
    """Synthetic camera diagnostics without a misleading real-image comparison column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not queries or any(q.get("split") != "novel" for q in queries):
        raise ValueError("Novel diagnostics need explicitly synthetic queries")
    fig, axes = plt.subplots(
        len(queries), len(renders), figsize=(5 * len(renders), 2.8 * len(queries)), squeeze=False
    )
    for i, query in enumerate(queries):
        for ax, (name, folder) in zip(axes[i], renders.items()):
            ax.imshow(image(folder / query["name"]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name if i == 0 else "", fontsize=12)
        key, value = next(iter(query["perturbation"].items()))
        label = f"Lateral {value:+g} m" if key == "side_m" else f"Yaw {value:+g}°"
        axes[i, 0].set_ylabel(label, fontsize=11)
    fig.suptitle("Synthetic offsets: no matching real photograph", fontsize=14)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def evaluate(manifest, queries, renders, out, *, device="cpu"):
    train = evaluation_training(manifest, queries)
    os.environ.setdefault("TORCH_HOME", str(out.resolve().parents[0] / "torch-cache"))
    import lpips
    import torch

    metric = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    torch.set_num_threads(6)
    rows = []
    for q in queries:
        reference = image(q["file_path"])
        if reference.shape != (760, 1344, 3):
            raise ValueError("Expected native 1344x760 evaluation image")
        tensor = torch.from_numpy(reference.copy()).permute(2, 0, 1)[None].to(device) * 2 - 1
        row = {"name": q["name"], "coverage": coverage(q, train), "scores": {}}
        for name, folder in renders.items():
            path = folder / q["name"]
            if not path.exists():
                row["scores"][name] = {"error": "missing render"}
                continue
            prediction = image(path)
            score = pixel_scores(reference, prediction)
            other = torch.from_numpy(prediction.copy()).permute(2, 0, 1)[None].to(device) * 2 - 1
            with torch.inference_mode():
                score["lpips_alex"] = float(metric(tensor, other).item())
            row["scores"][name] = score
        rows.append(row)
        print(q["name"], row["scores"], flush=True)
    result = {
        "device": device,
        "summary": summarize(rows, renders),
        "frames": rows,
        "coverage_groups": {
            "lateral_ge_1m": summarize(
                [r for r in rows if r["coverage"]["lateral_m"] >= 1], renders
            ),
            "heading_ge_10deg": summarize(
                [r for r in rows if r["coverage"]["heading_deg"] >= 10], renders
            ),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--render", action="append", required=True, help="name=directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()

    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    evaluate(
        read(args.manifest),
        read(args.queries),
        {k: Path(v) for k, v in (s.split("=", 1) for s in args.render)},
        args.out,
        device=args.device,
    )
    if args.figures:
        figures(
            read(args.manifest),
            read(args.queries),
            {k: Path(v) for k, v in (s.split("=", 1) for s in args.render)},
            args.figures,
        )


if __name__ == "__main__":
    main()
