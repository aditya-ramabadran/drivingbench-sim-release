"""Prospective driving health and post-shutdown evidence for A/C/D pilots."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import ExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from unittest.mock import patch

from drivingbench.replay.clients import DRIVING_SERVERS
from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.batch import save_new
from drivingbench.sim.hazard_evidence import driving_stream_paths, received_observations
from drivingbench.sim.studies.continuation import read
from drivingbench.sim.studies.pilot_exposure import _pair, verify_receipts

FAULT_KINDS = {"render_error", "sim_render_error", "runtime_error", "sim_runtime_error"}
_AUTH = re.compile(
    r"invalid_api_key|authentication_error|invalid x-api-key|not logged in|"
    r"oauth token (?:has )?(?:expired|expires soon)",
    re.IGNORECASE,
)
_CAPACITY = re.compile(
    r"usage_limit_reached|rate_limit_error|insufficient_quota|overloaded_error|"
    r"429.{0,40}too many requests|"
    r"Selected model is at capacity\. Please try a different model\.",
    re.IGNORECASE,
)


def _provider(errors):
    found = {
        kind
        for kind, pattern in (("failed_auth", _AUTH), ("failed_capacity", _CAPACITY))
        if any(pattern.search(e) for e in errors)
    }
    if len(found) != 1 or any(
        not (_AUTH.search(e) or _CAPACITY.search(e) or re.fullmatch(r"client_exit_-?\d+", e))
        for e in errors
    ):
        return None
    return found.pop()


def _failure_provider(error):
    if isinstance(error, str):
        errors = error.split("; ")
    elif isinstance(error, list) and all(isinstance(e, str) for e in error):
        errors = error
    else:
        return None
    return _provider(errors)


def _lines(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Non-object evidence in {path}")
    return rows


def _alive(thread):
    return thread.is_alive() if thread is not None else False


@contextmanager
def observe_lifecycle(directory, renderer_factory):
    """Observe the same app and renderer without changing startup or cleanup ownership."""
    directory = Path(directory)
    evidence = dict(app_created=False, entered=False, exited=False, renderers=[])
    original_app = study.simulator_app
    with ExitStack() as patches:

        def app_factory(*args, **kwargs):
            app = original_app(*args, **kwargs)
            evidence["app_created"] = True
            original_lifespan = app.router.lifespan_context

            @asynccontextmanager
            async def lifespan(application):
                try:
                    async with original_lifespan(application):
                        evidence["entered"] = True
                        yield
                except BaseException as error:
                    evidence["lifespan_error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    evidence["exited"] = True
                    state = app.state
                    for name, attr, thread in (
                        ("native", "sim_native", "_thread"),
                        ("cameras", "sim_cameras", "thread"),
                        ("recording", "recording", "thread"),
                        ("episode", "sim_episode", "_worker"),
                    ):
                        obj = getattr(state, attr, None)
                        evidence[name] = dict(
                            present=obj is not None,
                            alive=_alive(getattr(obj, thread, None)),
                            error=getattr(obj, "error", None),
                        )

            app.router.lifespan_context = lifespan
            return app

        def factory(*args, **kwargs):
            renderer = renderer_factory(*args, **kwargs)
            record = dict(
                close_called=False,
                closed=False,
                close_calls=[],
                process_alive=False,
                reader_alive=False,
            )
            processes, readers = [], []
            evidence["renderers"].append(record)
            original_close = renderer.close

            def close():
                worker = getattr(renderer, "worker", None)
                process = getattr(worker, "process", None)
                reader = getattr(worker, "reader", None)
                if reader is not None and reader not in readers:
                    readers.append(reader)
                if process is not None and process not in processes:
                    processes.append(process)
                call = dict(pid=getattr(process, "pid", None))
                record["close_calls"].append(call)
                record["close_called"] = True
                try:
                    return original_close()
                except BaseException as error:
                    record["error"] = call["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    current = getattr(worker, "process", None)
                    if current is not None and current not in processes:
                        processes.append(current)
                    call.update(
                        process_alive=any(p.poll() is None for p in processes),
                        reader_alive=any(_alive(r) for r in readers),
                    )
                    record.update(
                        closed=True,
                        process_alive=record["process_alive"] or call["process_alive"],
                        reader_alive=record["reader_alive"] or call["reader_alive"],
                    )

            patches.enter_context(patch.object(renderer, "close", close))
            return renderer

        patches.enter_context(patch.object(study, "simulator_app", app_factory))
        try:
            yield factory
        except BaseException as error:
            evidence["cell_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            save_new(directory / "lifecycle.json", evidence)


def _closed(evidence):
    problems = []
    if not all(evidence.get(k) is True for k in ("app_created", "entered", "exited")):
        problems.append("producer_lifecycle_incomplete")
    if evidence.get("cell_error") or evidence.get("lifespan_error"):
        problems.append("producer_cleanup_or_cell_error")
    for name in ("native", "cameras", "recording", "episode"):
        row = evidence.get(name, {})
        if (
            row.get("present") is not True
            or row.get("alive") is not False
            or row.get("error")
            or (row.get("error") is not None and not isinstance(row["error"], str))
        ):
            problems.append(f"{name}_closure_fault")
    renderers = evidence.get("renderers", [])
    if not renderers or any(
        r.get("close_called") is not True
        or r.get("closed") is not True
        or r.get("process_alive") is not False
        or r.get("reader_alive") is not False
        or r.get("error")
        for r in renderers
    ):
        problems.append("renderer_closure_fault")
    return problems


def _receipts(directory, events, servers, *, provider, supervisor):
    path = directory / "live-receipts.jsonl"
    receipts = _lines(path) if path.exists() or not provider else []
    audit = read(directory / "hazard-evidence.json")
    verified = verify_receipts(receipts, audit, events)
    problems, count = [], 0
    if type(supervisor.get("confirmed_calls")) is not int or supervisor["confirmed_calls"] != len(
        receipts
    ):
        problems.append("receipt_supervisor_count_mismatch")
    if any(r.get("supervisor_error") for r in receipts):
        problems.append("receipt_supervisor_error")
    rejection = directory / "live-receipt-rejections.jsonl"
    if rejection.exists() and _lines(rejection):
        problems.append("receipt_correspondence_rejected")
    observations = audit["receipt"]["observations"]
    attempt = read(directory / "attempt.json")
    stream = Path(attempt.get("driving_stream", directory / "client/stdout.jsonl"))
    if (
        stream != directory / "client/stdout.jsonl"
        and stream != directory / "driving-streams.jsonl"
    ):
        raise ValueError("Driving stream must belong to this exact attempt")
    paths = driving_stream_paths(stream)
    for source in paths:
        if source.exists():
            _lines(source)
        elif not provider or any(r.get("kind") == "sim_observe_delivery" for r in events):
            raise ValueError("Missing public driving stream")
    public = received_observations(stream, servers)
    if read(directory / "exposure-pending.json"):
        problems.append("unresolved_producer_selection")
    if len(receipts) != len(public) or len(observations) != len(public):
        problems.append("unresolved_received_observation")
    for receipt in verified:
        _, errors = _pair(receipt)
        matches = [r for r in public if r["call_id"] == receipt.get("call_id")]
        if len(matches) != 1 or len(matches[0]["images"]) != 2:
            errors.append("missing_public_paired_receipt")
        else:
            frames = sorted(receipt["frames"], key=lambda f: ("narrow", "wide").index(f["camera"]))
            if [hashlib.sha256(raw).hexdigest() for raw in matches[0]["images"]] != [
                f.get("received_sha256") for f in frames
            ] or any(f.get("timestamp") != matches[0]["status"]["timestamp"] for f in frames):
                errors.append("public_receipt_bytes_or_timestamp_mismatch")
        problems.extend(errors)
        count += not errors
    return count, problems


def driving_health(cell, output, *, servers=DRIVING_SERVERS):
    """Faults override provider labels; post-drive reporting does not invalidate clean driving."""
    folder = Path(output) / cell["assignment"]["id"]
    directory = folder / "attempt-1"
    problems, paired, provider = [], 0, None
    try:
        lifecycle = read(folder / "lifecycle.json")
        if not directory.exists():
            auth = read(folder / "auth-status.json")
            skip = cell.get("skipped")
            credential_failure = False
            if skip == "credential_preparation_failed" and auth.get("eligible") is True:
                failures = [read(folder / f"credential-attempt-{i}.json") for i in (1, 2)]
                credential_failure = all(
                    isinstance(r.get("error"), str) and _AUTH.search(r["error"]) for r in failures
                )
            if (
                cell.get("status") == "failed_auth"
                and (
                    (skip == "auth_or_token_lifetime_unavailable" and auth.get("eligible") is False)
                    or credential_failure
                )
                and lifecycle == dict(app_created=False, entered=False, exited=False, renderers=[])
            ):
                provider = "failed_auth"
            else:
                problems.append("unproven_preflight_failure")
        else:
            problems.extend(_closed(lifecycle))
            read(directory / "dispatch.json")
            attempt = read(directory / "attempt.json")
            episode = read(directory / "episode/summary.json")
            events = _lines(directory / "episode/events.jsonl")
            if (
                cell.get("attempts") != [attempt]
                or attempt.get("episode") != episode
                or attempt.get("assignment") != cell["assignment"]
                or attempt.get("status") != cell.get("status")
            ):
                problems.append("returned_or_closed_attempt_mismatch")
            errors = attempt["client_errors"]
            if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
                raise ValueError("Malformed client errors")
            if attempt.get("stop_error") is not None and not isinstance(attempt["stop_error"], str):
                raise ValueError("Malformed stop error")
            provider = _provider(errors)
            if provider != cell.get("status"):
                provider = None
            if attempt.get("stop_error"):
                problems.append("native_stop_error")
            if (
                episode.get("stopped") is not True
                or not isinstance(episode.get("native_end_reason"), str)
                or not episode["native_end_reason"]
            ):
                problems.append("native_stop_unproven")
            if (
                type(episode.get("unqueued_after_writer_error")) is not int
                or episode["unqueued_after_writer_error"] != 0
            ):
                problems.append("episode_writer_fault")
            for row in [episode.get("infrastructure_failure"), *events]:
                if not row:
                    continue
                kind = row.get("kind", "infrastructure_failure")
                if (
                    kind in FAULT_KINDS
                    or (kind == "sim_frame" and row.get("camera_errors"))
                    or (
                        kind == "infrastructure_failure"
                        and (
                            not provider
                            or row.get("source") != "model_client"
                            or _failure_provider(row.get("error")) != provider
                            or row.get("braking_error")
                        )
                    )
                ):
                    problems.append(f"structured_driving_fault:{kind}")
            supervisor = read(directory / "receipt-supervisor.json")
            if not isinstance(supervisor.get("censored"), list):
                problems.append("malformed_receipt_supervisor_censored")
            elif supervisor["censored"]:
                problems.append("receipt_supervisor_censored")
            paired, receipt_problems = _receipts(
                directory, events, servers, provider=provider, supervisor=supervisor
            )
            problems.extend(receipt_problems)
            if any(e.startswith(("mcp_tool_failure:", "mcp_permission_denials:")) for e in errors):
                problems.append("driving_mcp_failure")
            if not provider and (errors or not attempt.get("session_id")):
                problems.append("unclassified_client_failure")
            if cell.get("status") == "complete":
                if not paired or provider or episode.get("infrastructure_failure"):
                    problems.append("no_proven_healthy_paired_drive")
            elif not provider:
                problems.append("noncomplete_drive_without_provider_proof")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        problems.append(f"missing_or_invalid_evidence:{type(error).__name__}")
    return dict(
        infrastructure_problem=bool(problems),
        problems=list(dict.fromkeys(problems)),
        provider_failure=provider,
        model_unavailable=provider == "failed_auth" and not problems,
        verified_pairs=paired,
    )
