"""Isolated visual-environment entry point for the warm scene renderers."""

import argparse
import copy
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from drivingbench.sim.object_worker import emit, serve_requests
from drivingbench.sim.persistent import PersistentRenderer
from drivingbench.sim.wide_renderer import WideRenderer


def run(scene_path, output, stream):
    scene = json.loads(Path(scene_path).read_text())
    with ExitStack() as cleanup:
        renderers, readiness = {}, {}
        factories = {"narrow": PersistentRenderer}
        if "wide_camera" in scene:
            factories["wide"] = WideRenderer
        for camera, factory in factories.items():
            renderer = factory(scene_path, output=Path(output) / camera, new_session=False)
            cleanup.callback(renderer.close)
            renderers[camera] = renderer
            readiness[camera] = renderer.start()
        executor = cleanup.enter_context(ThreadPoolExecutor(max_workers=len(renderers)))
        emit(dict(readiness["narrow"], cameras=readiness))

        def render(request):
            started = time.perf_counter()
            directory = Path(request["output"])
            directory.mkdir(parents=True, exist_ok=True)
            paths, errors, metrics = {}, {}, {}
            pending = {
                camera: executor.submit(
                    renderer.render,
                    request["pose"],
                    sim_time_s=request.get("sim_time_s", 0),
                    actor_onsets=request.get("actor_onsets", {}),
                    **(
                        {"actor_states": request["actor_states"]}
                        if "actor_states" in request
                        else {}
                    ),
                )
                for camera, renderer in renderers.items()
            }
            for camera, future in pending.items():
                renderer = renderers[camera]
                try:
                    jpeg = future.result()
                    path = directory / f"{camera}.jpg"
                    path.write_bytes(jpeg)
                    paths[camera] = str(path)
                    if camera == "narrow":
                        metrics.update(copy.deepcopy(renderer.last_metrics))
                    else:
                        metrics[camera] = copy.deepcopy(renderer.last_metrics)
                except Exception as error:
                    errors[camera] = f"{type(error).__name__}: {error}"
            metrics["elapsed_s"] = time.perf_counter() - started
            return {"paths": paths, "errors": errors, "metrics": metrics}

        serve_requests(stream, render)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def terminate(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    run(args.scene, args.output, sys.stdin)


if __name__ == "__main__":
    main()
