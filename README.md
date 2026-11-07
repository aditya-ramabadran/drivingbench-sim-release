# DrivingBench simulator and offline testbeds

Code for *How Frontier Models Drive Around Humans: From a Real Car to a Validated Simulator*
(Aditya Ramabadran, Simon Mahns, Tobias Gessler; HumanRLI Workshop at CoRL 2026). It contains the
DrivingBench harness as used for the Toyota Corolla runs (the harness itself is at
<https://github.com/aditya-ramabadran/drivingbench_harness_code>, release `v1.0`) together with the
closed-loop simulator of the Corolla's parking lot and the replay and fork testbeds used in the
paper. The newer Tesla simulator ships with the harness's `v1.1` release. Project page:
<https://drivingbench.com>.

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

## Not included

- The trained Gaussian-splat checkpoint and the raw camera footage it was trained on. Every other
 scene file is included; `sim/visual_splat.py` retrains the splat from footage of any site
 (set `DRIVINGBENCH_SITE_ORIGIN="lat,lon"` for a site other than the paper's).

## License

MIT (`LICENSE`). `vendor/openpilot` and `native/` keep openpilot's MIT license.
