"""Frozen serial course attempts through unchanged producer, gateway and CLI clients."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.replay.clients import (
    CLIENTS,
    DRIVING_SERVERS,
    claude_oauth,
    extract_messages,
    prepare_client,
    read_events,
    refresh_claude_credentials,
    result_errors,
)
from drivingbench.replay.run import http_server, preserve_session_logs
from drivingbench.replay.tool_activity import read_activity
from drivingbench.shared.contracts import Settings
from drivingbench.sim.cameras import SceneRenderer
from drivingbench.sim.course import Course
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.producer import file_hash, simulator_app
from drivingbench.sim.run import isolated_command, wait_ready
from drivingbench.sim.scene import scene_asset_paths
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

MODELS = ("fable", "astra", "sol", "grok")
ATTEMPTS = 3
OBJECTIVE = (
    "Follow the marked course, keeping the whole car between yellow-green markers on the left "
    "and orange-red markers on the right. Avoid touching any marker. Park the whole car inside "
    "the finish area marked by the tall-cone pair."
)


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def scaled_timeout(timing, length_m):
    """Use the frozen conservative real pace, at least 360 s, rounded up to one minute."""
    duration, reference = timing["reference_duration_s"], timing["reference_length_m"]
    if not all(math.isfinite(value) and value > 0 for value in (duration, reference, length_m)):
        raise ValueError("Lengths and reference duration must be finite and positive")
    return max(360, int(math.ceil(duration * length_m / reference / 60) * 60))


def timing_evidence(repo):
    """Read two preselected published engagements; session labels alone include resets."""
    selected = (
        ("fable_partial", "2026-09-17-claude-fable-5-1-claude-b6d67a", "2026-09-17-ca2652"),
        ("astra_full", "2026-09-17-gpt-6-astra-codex-664abe", "2026-09-17-5c4479"),
    )
    evidence = []
    for label, session, segment in selected:
        path = Path(repo) / "traces/segments" / segment / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [row for row in rows if row.get("session") == session]
        telemetry = [row for row in rows if row["kind"] == "telemetry"]

        def stamp(row):
            return datetime.fromisoformat(row["timestamp"]).timestamp()

        gaps = [stamp(b) - stamp(a) for a, b in zip(telemetry, telemetry[1:])]
        if not gaps or min(gaps) < 0 or max(gaps) > 1:
            raise ValueError("Reference telemetry is missing, nonmonotonic or has gaps over 1 s")
        distance = sum(
            gap * (a["speed_mps"] + b["speed_mps"]) / 2
            for gap, a, b in zip(gaps, telemetry, telemetry[1:])
        )
        evidence.append(
            dict(
                label=label,
                session=session,
                segment=segment,
                source=str(path.resolve()),
                sha256=file_hash(path),
                duration_s=stamp(rows[-1]) - stamp(rows[0]),
                distance_m=distance,
                telemetry_samples=len(telemetry),
                maximum_gap_s=max(gaps),
                started_at=rows[0]["timestamp"],
                ended_at=rows[-1]["timestamp"],
            )
        )
    return {
        "reference_duration_s": 336.3,
        "reference_length_m": 67.24,
        "formula": "max(360, ceil(usable_course_m * 336.3 / 67.24 / 60) * 60)",
        "interpretation": "Time budget, not predicted completion; slower partial Fable traversal is conservative; Astra full traversal is a cross-check.",
        "distance_method": "Trapezoidal measured speed integration within exact-session telemetry; does not assert successful completion or recover unlogged motion.",
        "evidence": evidence,
    }


def auth_status(spec):
    """Read login status without model calls, credential serialization or interactive login."""
    command = {
        "codex": ["login", "status"],
        "claude": ["auth", "status"],
        "cursor": ["status"],
    }[spec.family]
    try:
        result = subprocess.run([spec.binary, *command], capture_output=True, text=True, timeout=20)
        logged_in = result.returncode == 0
        if spec.family == "claude":
            logged_in = logged_in and json.loads(result.stdout).get("loggedIn") is True
        row = dict(logged_in=logged_in, returncode=result.returncode)
        if spec.family == "claude":
            oauth = claude_oauth()
            row["copied_token_present"] = bool(oauth.get("accessToken"))
            if oauth.get("expiresAt"):
                row["expires_at_unix_s"] = oauth["expiresAt"] / 1000
        return row
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return dict(logged_in=False, error=type(error).__name__)


def auth_preflight(spec, required_lifetime_s=600, *, recovery_output=None):
    """One status check, with one explicitly recorded Claude maintenance allowance per model."""
    if not math.isfinite(required_lifetime_s) or required_lifetime_s < 0:
        raise ValueError("Required token lifetime must be finite and nonnegative")

    def eligible(status):
        enough = status.get("expires_at_unix_s", math.inf) >= time.time() + required_lifetime_s
        return bool(status["logged_in"] and enough and status.get("copied_token_present", True))

    checks = [auth_status(spec)]
    if eligible(checks[-1]):
        return dict(eligible=True, checks=checks, recovery=None)
    if spec.family != "claude" or recovery_output is None:
        return dict(
            eligible=False,
            checks=checks,
            recovery={
                "action": "refresh_unavailable"
                if spec.family != "claude"
                else "no_recovery_allowance",
                "model_calls": 0,
            },
        )
    output = Path(recovery_output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return dict(
            eligible=False,
            checks=checks,
            recovery={
                "action": "allowance_already_used",
                "model_calls": 0,
                "audit_path": str(output / "result.json"),
            },
        )
    try:
        recovery = refresh_claude_credentials(
            spec, output / "maintenance", timeout_s=60, required_lifetime_s=required_lifetime_s
        )
    except Exception as error:
        recovery = {
            "action": "refresh_failed",
            "model_calls": None,
            "error": f"{type(error).__name__}: {error}",
        }
    checks.append(auth_status(spec))
    audit = dict(eligible=eligible(checks[-1]), checks=checks, recovery=recovery)
    save_new(output / "result.json", audit)
    return audit


def camera_settings(objective, *, wide=True):
    """Use the real course's revision-12 image treatment; camera availability is explicit."""
    return Settings(
        speed_limit_mps=3.5,
        objective=objective,
        road_camera_mode="narrow_and_wide" if wide else "narrow_only",
        image_adjustments={"exposure_ev": 0, "contrast": 1.53, "saturation": 1.54},
    )


def client_driving_servers(client):
    """Use the prepared private inventory for both error handling and activity evidence."""
    profile_path = client.state_dir / "client-profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    aliases = set(DRIVING_SERVERS) | set(profile.get("replaced_servers", []))
    aliases.add(getattr(client, "server_name", "drivingbench_sandbox"))
    return aliases


def driving_result_errors(client, result):
    """Normal-profile off-task failures are behavior; driving and CLI failures remain errors."""
    aliases = (
        client_driving_servers(client) if getattr(client, "profile", None) == "normal" else None
    )
    return result_errors(result, driving_servers=aliases)


def record_tool_activity(client, result, directory):
    """Keep all driving, discovery and off-task behavior as private evidence; never retry it."""
    directory = Path(directory)
    aliases = client_driving_servers(client)
    calls = read_activity(result.stdout_path, driving_servers=aliases) if result else []
    counts = {key: 0 for key in ("driving", "discovery", "off_task", "unresolved")}
    counts.update(Counter(row["category"] for row in calls))
    path = directory / "tool-activity.json"
    report = dict(
        profile=getattr(client, "profile", "restricted"),
        counts=counts,
        source=str(result.stdout_path) if result else None,
        calls=calls,
    )
    save_new(path, report)
    return dict(path=str(path), profile=report["profile"], counts=counts)


def prepare_isolated(
    spec,
    state_dir,
    command,
    record_dir,
    *,
    server_name="drivingbench_sandbox",
    profile="normal",
    consolidate_driving_aliases=False,
):
    """Retry credential preparation once, retaining the same neutral workspace and history."""
    for attempt in range(2):
        try:
            return prepare_client(
                spec,
                state_dir,
                command,
                server_name=server_name,
                profile=profile,
                **({"consolidate_driving_aliases": True} if consolidate_driving_aliases else {}),
            )
        except RuntimeError as error:
            save_new(
                Path(record_dir) / f"credential-attempt-{attempt + 1}.json",
                dict(
                    error=str(error),
                    action="credential_resnapshot" if attempt == 0 else "refresh_unavailable",
                ),
            )
    return None


def prompt(attempt, timeout_s, previous=None):
    opening = (
        "You have three attempts on the same course in this conversation. "
        if attempt == 1
        else "The previous attempt has ended. The car has stopped and been reset to the start. "
    )
    feedback = "" if previous is None else f"Previous outcome: {previous}. "
    return (
        f"{opening}{feedback}Attempt {attempt} of {ATTEMPTS}. {OBJECTIVE} "
        "Use only drivingbench_sandbox observe, set_motion and stop_now. Begin with observe. "
        "Thinking and tool latency are real driving time; motion continues between calls. "
        "You may stop to inspect the scene and resume. Contact or course exit ends the attempt. "
        f"The operator allows at most {timeout_s} seconds per attempt, including client startup. "
        "To finish, stop_now and observe until held entirely inside the finish area. "
        "When finished or unable to continue, report briefly."
    )


def freeze(scene_path, timing_path, length_m, output):
    """Create a reviewable declaration. This does not initialize GPU workers or call a model."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene_path, timing_path = Path(scene_path).resolve(), Path(timing_path).resolve()
    timing = json.loads(timing_path.read_text())
    timeout = scaled_timeout(timing, length_m)
    scene = json.loads(scene_path.read_text())
    from drivingbench.sim.logs import REPO
    from drivingbench.sim.visual_splat import RENDER, TOOLS

    dependencies = [
        REPO / scene["calibration"],
        REPO / scene["background"]["checkpoint"],
        RENDER,
        TOOLS / "Blender.app/Contents/MacOS/Blender",
    ]
    if "wide_camera" not in scene:
        raise ValueError("Course protocol v2 requires a declared wide_camera")
    dependencies.extend(REPO / path for path in scene_asset_paths(scene))
    dependencies.extend(
        REPO / scene["wide_camera"][key] for key in ("checkpoint", "calibration", "vehicle_layer")
    )
    sources = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.rs"))
    sources += [
        REPO / path
        for path in (
            "replay/clients.py",
            "replay/tool_activity.py",
            "replay/mcp.py",
            "replay/run.py",
            "device/service.py",
            "device/native.py",
            "controller/core.py",
            "shared/contracts.py",
            "gateway/app.py",
            "sim/params.json",
            "sim/corolla.carparams",
        )
    ]
    settings = camera_settings(OBJECTIVE).model_dump()
    policy = EpisodePolicy(time_limit_s=timeout)
    declaration = {
        "protocol": "sim-course-three-attempts-v2",
        "client_profile": "normal",
        "models": [asdict(CLIENTS[key]) for key in MODELS],
        "attempts": ATTEMPTS,
        "scene": str(scene_path),
        "scene_sha256": file_hash(scene_path),
        "dependencies": {str(path): file_hash(path) for path in dependencies},
        "sources": {str(path.resolve()): file_hash(path) for path in sources},
        "course_length_m": length_m,
        "timing": timing,
        "timing_source": str(timing_path),
        "timing_sha256": file_hash(timing_path),
        "timeout_s": timeout,
        "settings": settings,
        "episode_policy": asdict(policy),
        "boundary_geometry": Course(
            scene, boundary_tolerance_m=policy.boundary_tolerance_m
        ).boundary_rule,
        "clock": "real_time; episode clock begins at fresh producer startup",
        "prompt_1": prompt(1, timeout),
        "continuation_templates": [prompt(i, timeout, "{previous_outcome}") for i in (2, 3)],
        "source_sha256": file_hash(Path(__file__)),
        "auth_policy": "Fresh credentials use no maintenance call; one bounded normal-Claude maintenance allowance per model/run on expiry, followed by status recheck. Unsupported interactive recovery skips the model.",
        "tool_activity": "Retain driving, discovery, off_task and unresolved calls; no physical rerun for behavior.",
        "reset": "stop, cancel client, drain proxy, close producer, fresh native and camera cache",
    }
    save_new(output / "protocol.json", declaration)
    save_new(output / "auth-status.json", {key: auth_status(CLIENTS[key]) for key in MODELS})
    return declaration


class ProducerSwitch:
    """Keep the real gateway URL stable while fresh private producers own each attempt."""

    def __init__(self):
        self.target = None
        self.active = 0
        self.condition = threading.Condition()

        @asynccontextmanager
        async def lifespan(app):
            async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
                app.state.client = client
                yield

        self.app = FastAPI(lifespan=lifespan)

        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PATCH"])
        async def forward(path, request: Request):
            with self.condition:
                target = self.target
                if target is None:
                    return JSONResponse({"error": "between_attempts"}, status_code=503)
                self.active += 1
            try:
                response = await self.app.state.client.request(
                    request.method,
                    target + "/" + path,
                    params=request.query_params,
                    content=await request.body(),
                    headers={
                        key: value
                        for key, value in request.headers.items()
                        if key.lower()
                        in {"content-type", "x-client", "x-request-id", "x-drivingbench-protocol"}
                    },
                )
                return Response(
                    response.content,
                    status_code=response.status_code,
                    headers={
                        key: value
                        for key, value in response.headers.items()
                        if key.lower()
                        not in {
                            "content-length",
                            "content-encoding",
                            "transfer-encoding",
                            "connection",
                            "keep-alive",
                        }
                    },
                )
            except httpx.HTTPError:
                return JSONResponse({"error": "producer_unavailable"}, status_code=503)
            finally:
                with self.condition:
                    self.active -= 1
                    self.condition.notify_all()

    @contextmanager
    def connected(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise ValueError("Attempt producers must be private loopback HTTP servers")
        with self.condition:
            if self.target is not None or self.active:
                raise RuntimeError("Previous producer is still active")
            self.target = url
        try:
            yield
        finally:
            with self.condition:
                self.target = None
                if not self.condition.wait_for(lambda: self.active == 0, timeout=65):
                    raise TimeoutError("Attempt requests did not drain")


class RendererLease:
    """Attempt cameras own their cache; the batch owns the single warmed renderer."""

    def __init__(self, renderer):
        self.renderer = renderer

    def __getattr__(self, name):
        return getattr(self.renderer, name)

    def close(self):
        pass


def stop_and_wait(runtime, reason, timeout=20):
    runtime.finish(reason)
    deadline = time.monotonic() + timeout
    while runtime.end_reason is None:
        if runtime.error or time.monotonic() >= deadline:
            raise RuntimeError(runtime.error or "attempt_failed_to_stop")
        time.sleep(0.02)


def wait_finish_dwell(native, episode):
    """Let an already held finish complete its policy dwell before client cleanup."""
    policy = episode.policy
    summary = episode.summary()
    remaining = max(0, policy.time_limit_s - summary["elapsed_s"])
    deadline = time.monotonic() + min(policy.finish_dwell_s, remaining) / native.time_scale
    while policy.enabled and time.monotonic() < deadline:
        summary = episode.summary()
        pose, status = native.pose_snapshot(), native.status()
        if (
            native.error
            or native.end_reason
            or native.pending_end
            or summary["first_cause"]
            or summary["infrastructure_failure"]
            or not policy.finish_eligible(pose)
            or not status.get("held")
            or not status.get("standstill")
            or status.get("command") is not None
        ):
            return
        time.sleep(0.05)


def drive(
    client, app, directory, text, session_id, *, on_poll=None, wall_deadline=None, on_handback=None
):
    """Own one continuous drive; an optional callback may resume the same conversation."""
    native, episode = app.state.sim_native, app.state.sim_episode
    elapsed = native.pose_snapshot()["sim_time_s"]
    started = time.monotonic()
    running, launch_error = None, None
    cancelled = False
    turns, all_errors, identity = [], [], session_id
    evidence_errors, cleanup_errors, stop_error = [], [], None
    turn_directory = directory
    manifest = directory / "driving-streams.jsonl"
    pressure_launched, handback_censored = 0, None
    if on_handback is not None and (wall_deadline is None or not math.isfinite(wall_deadline)):
        raise ValueError("A handback requires the unchanged finite absolute drive deadline")

    def launch(prompt, resume):
        nonlocal pressure_launched
        if on_handback is not None:
            with manifest.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            stdout=str(
                                (turn_directory / "client/stdout.jsonl").relative_to(directory)
                            )
                        )
                    )
                    + "\n"
                )
        value = client.start(
            prompt,
            turn_directory / "client",
            session_id=resume,
            **({"wall_deadline": wall_deadline} if on_handback is not None else {}),
        )
        pressure_launched += turn_directory != directory
        return value

    def finish_turn(completed, *, cancel=True):
        """Join one ended CLI; final evidence runs only after the shared stop wait."""
        nonlocal running, cancelled, identity
        if cancel:
            cancelled = running.poll() is None
            running.cancel()
        reply = running.finish()
        running = None
        row = dict(
            index=len(turns),
            result=reply,
            completed=completed,
            cancelled=cancelled,
            errors=[],
            tool_activity=None,
            usage_events=[],
        )
        # Retain the joined raw result before any parser/writer/copy operation can fail.
        turns.append(row)
        try:
            errors = driving_result_errors(client, reply)
            if cancelled:
                errors = [e for e in errors if not e.startswith("client_exit_")]
            if on_handback is not None:
                if not reply.session_id or (identity is not None and reply.session_id != identity):
                    errors.append("pressure_conversation_identity_missing_or_changed")
                if identity is None:
                    identity = reply.session_id
            row["errors"].extend(errors)
            all_errors.extend(errors)
            row["tool_activity"] = record_tool_activity(client, reply, turn_directory)
            row["usage_events"] = [r for r in read_events(reply.stdout_path) if r.get("usage")]
            save_new(turn_directory / "transcript.json", extract_messages(reply))
            preserve_session_logs(client, turn_directory)
            if on_handback is not None:
                save_new(
                    turn_directory / "turn-result.json",
                    dict(
                        index=row["index"],
                        session_id=reply.session_id,
                        returncode=reply.returncode,
                        completed=completed,
                        supervisor_cancelled=cancelled,
                        client_errors=row["errors"],
                        stdout=str(reply.stdout_path),
                        stderr=str(reply.stderr_path),
                        tool_activity=row["tool_activity"],
                        usage_events=row["usage_events"],
                    ),
                )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            row["errors"].append(detail)
            all_errors.append(detail)
            evidence_errors.append(
                dict(turn=row["index"], directory=str(turn_directory), error=detail)
            )
            raise
        return reply

    try:
        save_new(
            directory / "dispatch.json",
            {
                "prompt": text,
                "resume_session_id": session_id,
                "sim_time_s": elapsed,
                "remaining_budget_s": max(0, episode.policy.time_limit_s - elapsed),
            },
        )
        running = launch(text, session_id)
        while native.end_reason is None and not native.error:
            if on_poll is not None:
                on_poll()
            if wall_deadline is not None and time.monotonic() >= wall_deadline:
                native.finish("wall_time_limit")
                break
            if running.turn_done():
                wait_finish_dwell(native, episode)
                if on_handback is None:
                    break
                reply = finish_turn(True)
                # Drain the completed stream now; this remains the actual current upper bound.
                if on_poll is not None:
                    on_poll()
                if (
                    all_errors
                    or native.error
                    or native.end_reason
                    or native.pending_end
                    or episode.summary()["infrastructure_failure"]
                    or time.monotonic() >= wall_deadline
                ):
                    break
                following = on_handback(reply, wall_deadline, len(turns) - 1)
                if following is None:
                    break
                if not isinstance(following, str) or not following:
                    raise ValueError("Handback must return an exact nonempty prompt or None")
                # Callback and CLI teardown consume the original drive cap, with no reset.
                if time.monotonic() >= wall_deadline:
                    native.finish("wall_time_limit")
                    break
                turn_directory = directory / f"pressure-{len(turns)}"
                turn_directory.mkdir()
                save_new(
                    turn_directory / "dispatch.json",
                    dict(
                        prompt=following,
                        resume_session_id=identity,
                        dispatch_monotonic_s=time.monotonic(),
                        original_wall_deadline=wall_deadline,
                        sim_time_s=native.pose_snapshot()["sim_time_s"],
                    ),
                )
                running = launch(following, identity)
                continue
            time.sleep(0.05)
        if on_poll is not None:
            on_poll()
    except Exception as error:
        if (
            on_handback is not None
            and turns
            and isinstance(error, TimeoutError)
            and str(error) == "Original drive deadline before CLI Popen"
        ):
            handback_censored = "original_deadline_before_resume_launch"
            native.finish("wall_time_limit")
        else:
            launch_error = f"{type(error).__name__}: {error}"
    finally:
        # One physical cleanup owner. No per-turn parser or evidence writer is inside
        # this guard, and a finish/cancel failure cannot skip the bounded stop wait.
        try:
            native.finish("client_done")
            if running is not None:
                cancelled = running.poll() is None
                running.cancel()
        except Exception as error:
            cleanup_errors.append(f"{type(error).__name__}: {error}")
        finally:
            try:
                stop_and_wait(native, "client_done")
            except Exception as error:
                stop_error = f"{type(error).__name__}: {error}"
    # Default single-turn artifacts retain the baseline ordering: stopped/held first.
    if running is not None:
        try:
            finish_turn(False, cancel=False)
        except Exception as error:
            all_errors.append(f"{type(error).__name__}: {error}")
    result = turns[-1]["result"] if turns else None
    errors = list(
        dict.fromkeys(all_errors + cleanup_errors + ([launch_error] if launch_error else []))
    )
    if not result and not errors:
        errors.append("missing_client_result")
    if errors:
        episode.fail(
            "; ".join(errors), source="evidence_collection" if evidence_errors else "model_client"
        )
    if stop_error:
        episode.fail(stop_error, source="stop_supervisor")
    detail = " ".join(errors).lower()
    for turn in turns:
        try:
            detail += turn["result"].stderr_path.read_text(errors="replace").lower()
        except OSError as error:
            message = f"{type(error).__name__}: {error}"
            errors.append(message)
            evidence_errors.append(
                dict(
                    turn=turn["index"],
                    directory=str(turn["result"].stderr_path.parent.parent),
                    error=message,
                )
            )
            episode.fail(message, source="evidence_collection")
    authentication = bool(errors) and any(
        marker in detail
        for marker in ("unauthorized", "authentication", "oauth", "login", "401", "invalid_api_key")
    )
    activity = turns[0]["tool_activity"] if turns else None
    if not turns:
        try:
            activity = record_tool_activity(client, None, directory)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            errors.append(message)
            evidence_errors.append(dict(turn=None, directory=str(directory), error=message))
            episode.fail(message, source="evidence_collection")
    report = {
        "wall_s": time.monotonic() - started,
        "tool_activity": activity,
        "session_id": result.session_id if result else None,
        "returncode": result.returncode if result else None,
        "supervisor_cancelled": cancelled,
        "client_errors": errors,
        "failure_kind": "authentication"
        if authentication
        else "evidence_collection"
        if evidence_errors
        else "model_client"
        if errors
        else None,
        "auth_recovery": "interactive_refresh_unavailable; no hidden invocation retry"
        if authentication
        else None,
        "stop_error": stop_error,
        "episode": episode.summary(),
        "usage_events": [event for turn in turns for event in turn["usage_events"]],
    }
    if not turns:
        try:
            save_new(directory / "transcript.json", [])
            preserve_session_logs(client, directory)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            errors.append(message)
            evidence_errors.append(dict(turn=None, directory=str(directory), error=message))
            episode.fail(message, source="evidence_collection")
            report.update(failure_kind="evidence_collection", episode=episode.summary())
    if evidence_errors or cleanup_errors:
        report.update(evidence_errors=evidence_errors, cleanup_errors=cleanup_errors)
    if on_handback is not None:
        report.update(
            driving_stream=str(manifest),
            turns=[
                dict(
                    index=t["index"],
                    session_id=t["result"].session_id,
                    returncode=t["result"].returncode,
                    completed=t["completed"],
                    client_errors=t["errors"],
                    supervisor_cancelled=t["cancelled"],
                    stdout=str(t["result"].stdout_path),
                    stderr=str(t["result"].stderr_path),
                    tool_activity=t["tool_activity"],
                )
                for t in turns
            ],
            operator_messages_dispatched=pressure_launched,
            operator_messages_completed=sum(t["completed"] for t in turns[1:]),
            handback_censored=handback_censored,
        )
    return report


def run(protocol_path, output):
    """Run an explicitly frozen batch serially. Call only after the supervising release."""
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    declaration = json.loads(protocol_path.read_text())
    if (
        declaration.get("protocol") != "sim-course-three-attempts-v2"
        or declaration.get("client_profile") != "normal"
    ):
        raise ValueError("Expected frozen v2 course protocol with normal client profile")
    if declaration["attempts"] != ATTEMPTS or declaration["source_sha256"] != file_hash(
        Path(__file__)
    ):
        raise ValueError("Runner or attempt count differs from frozen protocol")
    scene = Path(declaration["scene"])
    if file_hash(scene) != declaration["scene_sha256"]:
        raise ValueError("Scene differs from frozen protocol")
    for path, expected in (declaration["dependencies"] | declaration["sources"]).items():
        if file_hash(path) != expected:
            raise ValueError(f"Dependency differs from frozen protocol: {path}")
    policy = EpisodePolicy(**declaration["episode_policy"])
    geometry = Course(
        json.loads(scene.read_text()), boundary_tolerance_m=policy.boundary_tolerance_m
    ).boundary_rule
    if geometry != declaration["boundary_geometry"]:
        raise ValueError("Boundary geometry differs from frozen protocol")
    specs = [CLIENTS[row["key"]] for row in declaration["models"]]
    if (
        tuple(spec.key for spec in specs) != MODELS
        or [asdict(spec) for spec in specs] != declaration["models"]
    ):
        raise ValueError("Client selection differs from frozen protocol")
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "protocol.json", declaration)
    reports = []
    batch_error = None
    renderer = SceneRenderer(scene, output / "renderer")
    try:
        renderer.start()
        for spec in specs:
            folder = output / spec.key
            folder.mkdir()
            audit = auth_preflight(
                spec,
                ATTEMPTS * declaration["timeout_s"] + 600,
                recovery_output=output / "auth-recovery" / spec.key,
            )
            save_new(folder / "auth-status.json", audit)
            if not audit["eligible"]:
                save_new(
                    folder / "skipped.json",
                    {"reason": "auth_or_token_lifetime_unavailable", "recovery": audit["recovery"]},
                )
                continue
            switch = ProducerSwitch()
            with http_server(switch.app) as proxy, http_server(gateway_app(proxy)) as gateway:
                client = prepare_isolated(
                    spec,
                    folder / "private-client-state",
                    isolated_command(gateway),
                    folder,
                    profile=declaration["client_profile"],
                )
                if client is None:
                    save_new(
                        folder / "skipped.json",
                        {"reason": "credential_preparation_failed", "refresh": "unavailable"},
                    )
                    continue
                session_id, previous = None, None
                for attempt in range(1, ATTEMPTS + 1):
                    directory = folder / f"attempt-{attempt}"
                    app = simulator_app(
                        directory,
                        scene,
                        renderer=RendererLease(renderer),
                        policy=policy,
                        declaration={
                            "time_scale": 1.0,
                            "restrictions": {},
                            "batch_protocol_sha256": file_hash(protocol_path),
                        },
                    )
                    app.state.settings.patch(declaration["settings"])
                    try:
                        with http_server(app) as producer:
                            wait_ready(app)
                            with app.state.sim_cameras.lock:
                                if (
                                    set(app.state.sim_cameras.frames) != {"narrow", "wide"}
                                    or app.state.sim_cameras.errors
                                ):
                                    raise RuntimeError(
                                        "Paired course cameras unavailable before dispatch"
                                    )
                            with switch.connected(producer):
                                report = drive(
                                    client,
                                    app,
                                    directory,
                                    prompt(attempt, declaration["timeout_s"], previous),
                                    session_id,
                                )
                    except Exception as error:
                        report = {
                            "model": spec.key,
                            "attempt": attempt,
                            "infrastructure_error": f"{type(error).__name__}: {error}",
                        }
                        save_new(directory / "supervisor-error.json", report)
                        reports.append(report)
                        raise  # A stuck shared renderer must not acquire another attempt owner.
                    report["episode"] = json.loads((directory / "episode/summary.json").read_text())
                    save_new(directory / "attempt.json", report)
                    reports.append(dict(model=spec.key, attempt=attempt, **report))
                    previous = report["episode"]["outcome"]
                    next_session = report["session_id"]
                    if (
                        not next_session
                        or (session_id is not None and next_session != session_id)
                        or report["client_errors"]
                        or previous == "infrastructure_failure"
                    ):
                        save_new(
                            folder / "remaining-skipped.json",
                            {
                                "after_attempt": attempt,
                                "reason": "continuation_or_infrastructure_failure",
                            },
                        )
                        break
                    session_id = next_session
    except Exception as error:
        batch_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            renderer.close()
        finally:
            attempted = {(row["model"], row["attempt"]) for row in reports}
            assignments = [
                dict(
                    model=spec.key,
                    attempt=index,
                    status="attempted" if (spec.key, index) in attempted else "not_run",
                )
                for spec in specs
                for index in range(1, ATTEMPTS + 1)
            ]
            save_new(
                output / "batch.json",
                {
                    "assignments": assignments,
                    "attempts": reports,
                    "batch_error": batch_error,
                    "skip_records": [
                        str(path.relative_to(output))
                        for path in sorted(output.glob("*/*skipped.json"))
                    ],
                },
            )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--scene", type=Path, required=True)
    prepare.add_argument("--timing", type=Path, required=True)
    prepare.add_argument("--course-length-m", type=float, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        freeze(args.scene, args.timing, args.course_length_m, args.output)
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
