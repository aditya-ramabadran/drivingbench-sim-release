# DrivingBench simulator and offline testbeds

Anonymous release for double-blind review. It contains the DrivingBench harness (also at
<https://anonymous.4open.science/r/drivingbench_harness_v1-EEBC/>) together with the closed-loop
simulator and the replay and fork testbeds used in the paper.

A model drives through the same MCP tools (`observe`, `set_motion`, `stop_now`), gateway, request
envelope and openpilot controllers as on the real car. Only the layers below them are simulated:

| Layer | Real car | Simulator |
|---|---|---|
| Tools, gateway, controller | `mcp/`, `gateway/`, `controller/` | same code |
| openpilot control | patched openpilot on the comma device | same code, run tick for tick by `SimNative` (`sim/car.py`) |
| Vehicle | Toyota Corolla | plant fitted to logged drives (`sim/plant.py`, `sim/params.json`) |
| Camera | road and wide cameras | Gaussian-splat scene plus billboard actors (`sim/visual_*.py`, `sim/billboards.py`) |

On held-out drives, replaying the logged tool calls through the simulator gives a median
within-window path RMS of 0.31 m (218 windows), and 0.96 m at 30 s (89 windows).

## Layout

- `mcp/ gateway/ device/ controller/ shared/ ui/`: the harness, unchanged from the real-car runs.
- `sim/`: plant, runtime, renderer interface, scenes (`site1-*.json`), hazard and fork scenarios,
 and study runners.
- `replay/`: the replay testbed (logged camera frames, virtual car), hazard protocol and objective
 (`replay/hazard_protocol.py`), fork videos (`replay/fork.py`) and judging.
- `vendor/openpilot/`: pinned upstream openpilot (see `docs/upstream-sources.json`); the two
 DrivingBench changes are also kept separately in `native/`.
- `docs/`: simulator dynamics, protocols, judging rubrics and course measurements.

## Run

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest -q                                   # about three minutes; no data or GPU needed
uv run python -m drivingbench.sim.fit              # refit the plant from route logs
uv run python -m drivingbench.sim.validate         # score it on a held-out route
uv run python -m drivingbench.sim.run serve --output runs/sim/serve
```

The fit and validation scripts read openpilot route logs from `runs/artifacts/routes`, which are
not part of this release; the fitted plant is in `sim/params.json`. `docs/sim-dynamics.md`
describes the plant and its validation. The live renderer needs a Gaussian-splat checkpoint (see
below), Brush (`sim/brush_render.rs`), Blender and a Metal GPU.

## Withheld for anonymity

- The trained Gaussian-splat checkpoint and the raw camera footage it was trained on, which show
 the test site. Every other scene file is included; `sim/visual_splat.py` retrains the splat
 from footage of any site.
- The site's GPS origin. Positions are in local metres; set `DRIVINGBENCH_SITE_ORIGIN="lat,lon"`
 to index new footage.
- One quoted sentence in the fork prompt (`replay/fork.py`) that would identify the project.

## License

MIT (`LICENSE`). `vendor/openpilot` and `native/` keep openpilot's MIT license.
