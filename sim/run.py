"""Start a private local simulator or one explicitly requested model smoke drive."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.replay.clients import CLIENTS, read_events
from drivingbench.replay.run import http_server
from drivingbench.sim.cameras import SceneRenderer
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.producer import DEFAULT_SCENE, simulator_app

PROMPT = """Drive the car using drivingbench_sandbox. The marked course has yellow-green markers
on the left and orange-red markers on the right. Keep the whole car between them and avoid
contact. Use only observe, set_motion and stop_now; thinking time is real driving time.
This is a short engineering smoke drive: first observe, choose one safe 5-second motion along
the course at 0.5–1.0 m/s, observe to check progress, then stop_now, observe until held, and
report what happened. Do not attempt the full course. The operator caps this drive at 60 seconds.
"""


def wait_ready(app, timeout=180):
    deadline = time.monotonic() + timeout
    while "narrow" not in app.state.sim_cameras.frames:
        if time.monotonic() >= deadline or app.state.sim_cameras.errors.get("narrow"):
            raise RuntimeError(f"Simulator camera not ready: {app.state.sim_cameras.errors}")
        time.sleep(0.05)


def isolated_command(gateway):
    return [sys.executable, "-m", "drivingbench.replay.mcp", "--gateway-url", gateway]


def smoke(app, gateway, output, model):
    from drivingbench.sim.batch import (
        auth_preflight,
        driving_result_errors,
        prepare_isolated,
        record_tool_activity,
        save_new,
    )

    spec = CLIENTS[model]
    auth = auth_preflight(spec, 600, recovery_output=output / "auth-recovery" / model)
    save_new(output / "auth-preflight.json", auth)
    client = None
    if auth["eligible"]:
        client = prepare_isolated(
            spec, output / "client-state", isolated_command(gateway), output, profile="normal"
        )
    if client is None:
        report = dict(
            model=spec.model,
            skipped="auth_unavailable",
            reason="credential_preparation_failed" if auth["eligible"] else "auth_preflight_failed",
            auth=auth,
            client_profile="normal",
        )
        save_new(output / "smoke.json", report)
        return report
    (output / "prompt.txt").write_text(PROMPT)
    started = time.monotonic()
    running = client.start(PROMPT, output / "conversation")
    capped = False
    cancelled = False
    try:
        while not running.turn_done():
            if time.monotonic() - started >= 60 or app.state.sim_native.end_reason:
                capped = True
                break
            time.sleep(0.1)
    finally:
        app.state.sim_native.finish("smoke_end")
        cancelled = running.poll() is None
        running.cancel()
    result = running.finish()
    deadline = time.monotonic() + 10
    while app.state.sim_native.pose_snapshot()["speed_mps"] >= 0.05:
        if time.monotonic() >= deadline:
            raise RuntimeError("Car failed to stop after smoke")
        time.sleep(0.02)
    errors = driving_result_errors(client, result)
    if cancelled:
        errors = [error for error in errors if not error.startswith("client_exit_")]
    if errors:
        app.state.sim_episode.fail("; ".join(errors), source="model_client")
    report = {
        "model": spec.model,
        "client_profile": "normal",
        "tool_activity": record_tool_activity(client, result, output),
        "wall_s": time.monotonic() - started,
        "capped": capped,
        "client_returncode": result.returncode,
        "client_errors": errors,
        "supervisor_cancelled": cancelled,
        "episode": app.state.sim_episode.summary(),
        "camera": app.state.sim_cameras.last_metrics,
        "usage_events": [e for e in read_events(result.stdout_path) if e.get("usage")],
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "smoke"))
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=CLIENTS, default="sol")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    renderer = SceneRenderer(args.scene, output / "renderer")
    try:
        print(json.dumps({"renderer": renderer.start()}), flush=True)
        from drivingbench.sim.batch import camera_settings

        scene = json.loads(args.scene.read_text())
        app = simulator_app(
            output,
            args.scene,
            renderer=renderer,
            settings=camera_settings(PROMPT, wide="wide_camera" in scene),
            policy=EpisodePolicy(time_limit_s=60 if args.mode == "smoke" else 360),
        )
        from drivingbench.sim.operator import attach_operator

        attach_operator(app)
        with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
            wait_ready(app)
            endpoints = {
                "operator": producer + "/sim",
                "producer": producer,
                "gateway": gateway,
                "mcpServers": {
                    "drivingbench_sandbox": {
                        "command": sys.executable,
                        "args": isolated_command(gateway)[1:],
                    }
                },
            }
            (output / "local.json").write_text(json.dumps(endpoints, indent=2) + "\n")
            print(json.dumps(endpoints, indent=2), flush=True)
            if args.mode == "smoke":
                smoke(app, gateway, output, args.model)
            else:
                try:
                    while True:
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    app.state.sim_native.finish("operator_end")
                    deadline = time.monotonic() + 10
                    while app.state.sim_native.pose_snapshot()["speed_mps"] >= 0.05:
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(0.02)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
