"""Resume undispatched neutral-study cells with bounded post-drive retries."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.batch import save_new
from drivingbench.sim.producer import file_hash

POLICY = {
    "version": "neutral-v21-postdrive-continuation-1",
    "postdrive_retries": 1,
    "missing_postdrive_stops_study": False,
    "driving_infrastructure_stops_study": True,
    "completed_cell_replacements": False,
}


def read(path):
    return json.loads(Path(path).read_text())


def pending(protocol, previous, old_results):
    """Accept only a closed allocation prefix with no unrecorded dispatch directories."""
    rows = protocol["assignments"]
    cells = previous["cells"]
    if [cell["assignment"] for cell in cells] != rows[: len(cells)]:
        raise ValueError("Existing cells are not an exact allocation prefix")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate assignment ids")
    for cell in cells:
        saved = Path(old_results) / cell["assignment"]["id"] / "session.json"
        if read(saved) != cell:
            raise ValueError("Saved session differs from the closed batch")
    remainder = rows[len(cells) :]
    if any((Path(old_results) / row["id"]).exists() for row in remainder):
        raise ValueError("An undispatched assignment already has evidence")
    return remainder


def freeze_addendum(protocol_path, old_results, output):
    """Freeze supervisor identity and reference old evidence without rewriting it."""
    protocol_path, old_results, output = map(
        lambda path: Path(path).resolve(), (protocol_path, old_results, output)
    )
    protocol = read(protocol_path)
    study.validate_protocol(protocol)
    gate = read(old_results / "checkpoint-round-1.json")
    if gate.get("continue_automatically") is not True:
        raise ValueError("Original eight-drive checkpoint did not pass")
    previous = read(old_results / "batch.json")
    rows = pending(protocol, previous, old_results)
    evidence = [old_results / "batch.json", old_results / "checkpoint-round-1.json"]
    evidence.extend(old_results / c["assignment"]["id"] / "session.json" for c in previous["cells"])
    evidence.extend(old_results / c["assignment"]["id"] / "health.json" for c in previous["cells"])
    evidence = [path for path in evidence if path.exists()]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve()]
    output.mkdir(parents=True, exist_ok=False)
    addendum = dict(
        policy=POLICY,
        protocol_path=str(protocol_path),
        protocol_sha256=file_hash(protocol_path),
        deadline_utc=protocol["deadline_utc"],
        previous_results=str(old_results),
        new_results=str(output),
        completed_ids=[c["assignment"]["id"] for c in previous["cells"]],
        assignments_remaining=rows,
        evidence={str(path): file_hash(path) for path in evidence},
        supervisor_sources={str(path): file_hash(path) for path in sources},
        original_checkpoint=gate,
    )
    save_new(output / "addendum.json", addendum)
    return addendum


def validate_addendum(addendum):
    if addendum["policy"] != POLICY:
        raise ValueError("Continuation policy changed")
    if file_hash(addendum["protocol_path"]) != addendum["protocol_sha256"]:
        raise ValueError("Original protocol changed")
    for path, digest in (addendum["evidence"] | addendum["supervisor_sources"]).items():
        if file_hash(path) != digest:
            raise ValueError(f"Continuation dependency changed: {path}")
    protocol = read(addendum["protocol_path"])
    study.validate_protocol(protocol)
    if protocol["deadline_utc"] != addendum["deadline_utc"]:
        raise ValueError("Absolute deadline changed")
    previous = read(Path(addendum["previous_results"]) / "batch.json")
    rows = pending(protocol, previous, addendum["previous_results"])
    if rows != addendum["assignments_remaining"] or addendum["completed_ids"] != [
        cell["assignment"]["id"] for cell in previous["cells"]
    ]:
        raise ValueError("Continuation allocation changed")
    return protocol


@contextmanager
def postdrive_policy(deadline, previous_recovery=None):
    """Patch only post-drive calls; keep original client context and driving behavior."""
    original_followup, original_coder = study.followup, study.code_belief
    classify = study.reporting_failure
    original_auth = study.auth_preflight

    def auth(spec, required_lifetime_s=600, *, recovery_output=None):
        prior = Path(previous_recovery) / spec.key if previous_recovery else None
        if prior and prior.exists():
            recovery_output = None
        return original_auth(spec, required_lifetime_s, recovery_output=recovery_output)

    def retry(invoke, folder, budget, failure):
        folder = Path(folder)
        replies = []
        for name in ("first", "retry"):
            remaining = min(budget, deadline - time.monotonic())
            if remaining <= 0:
                break
            path = folder / name
            try:
                reply = invoke(path, remaining)
            except Exception as error:
                reply = dict(errors=[f"{type(error).__name__}: {error}"], same_context=False)
            kind = failure(reply)
            reply.update(failure_kind=kind, missing=kind is not None)
            save_new(path / "supervisor-reply.json", reply)
            replies.append(dict(path=str(path), reply=reply))
            if not kind:
                break
        selected = (
            dict(replies[-1]["reply"])
            if replies
            else dict(
                errors=["absolute_deadline"],
                same_context=False,
                failure_kind="not_run_deadline",
                missing=True,
            )
        )
        selected.update(question_attempts=replies, automatic_retries=max(0, len(replies) - 1))
        save_new(folder / "supervisor-summary.json", selected)
        return selected

    def followup(client, text, folder, session_id, budget):
        return retry(
            lambda path, cap: original_followup(client, text, path, session_id, cap),
            folder,
            budget,
            lambda reply: (
                classify(reply) or (None if reply.get("text", "").strip() else "empty_answer")
            ),
        )

    def coder(text, folder, budget):
        return retry(
            lambda path, cap: original_coder(text, path, cap),
            folder,
            budget,
            lambda reply: None if reply.get("label") in study.BELIEF_LABELS else "coder_failure",
        )

    with (
        patch.object(study, "auth_preflight", auth),
        patch.object(study, "followup", followup),
        patch.object(study, "code_belief", coder),
        patch.object(study, "reporting_failure", lambda reply: None),
    ):
        yield


def amended_health(cell, protocol, output):
    """Retain driving and receipt guard faults, exclude missing post-drive answers."""
    with patch.object(study, "reporting_failure", lambda reply: None):
        return study.checkpoint([cell], protocol, output)


def run(addendum_path, *, renderer_factory=study.SceneRenderer, cell_runner=study._cell):
    addendum_path = Path(addendum_path).resolve()
    addendum = read(addendum_path)
    protocol = validate_addendum(addendum)
    output = Path(addendum["new_results"])
    if any(
        (output / name).exists() for name in ("batch.json", "progress.jsonl", "dispatch-start.json")
    ):
        raise ValueError("Continuation already dispatched; do not restart it")
    if any((output / row["id"]).exists() for row in addendum["assignments_remaining"]):
        raise ValueError("Continuation contains unrecorded dispatch evidence")
    save_new(
        output / "dispatch-start.json",
        dict(addendum_sha256=file_hash(addendum_path), started_at=time.time()),
    )
    supervisor_hash = file_hash(addendum_path)
    deadline = (
        time.monotonic()
        + datetime.fromisoformat(addendum["deadline_utc"]).timestamp()
        - time.time()
    )
    cells, failure = [], None
    try:
        for row in addendum["assignments_remaining"]:
            if time.monotonic() >= deadline:
                break
            protocol = validate_addendum(addendum)
            if file_hash(addendum_path) != supervisor_hash:
                raise ValueError("Frozen supervisor addendum changed")
            folder = output / row["id"]
            try:
                with postdrive_policy(
                    deadline, Path(addendum["previous_results"]) / "auth-recovery"
                ):
                    cell = cell_runner(
                        protocol,
                        row,
                        folder,
                        output / "auth-recovery",
                        deadline,
                        renderer_factory=renderer_factory,
                    )
                for key in protocol["followup_order"]:
                    answer = cell.get(key)
                    if answer:
                        # The frozen cell annotates its classifier result after the wrapper.
                        answer["failure_kind"] = (
                            answer["question_attempts"][-1]["reply"]["failure_kind"]
                            if answer.get("question_attempts")
                            else study.reporting_failure(answer)
                        )
                cell["missing_postdrive"] = [
                    key
                    for key in protocol["followup_order"]
                    if not cell.get(key) or cell[key].get("missing")
                ]
                cell["supervisor_policy"] = POLICY["version"]
            except Exception as error:
                cell = dict(
                    assignment=row,
                    status="infrastructure_failure",
                    attempts=[],
                    error=f"{type(error).__name__}: {error}",
                )
            cells.append(cell)
            save_new(folder / "session.json", cell)
            health = amended_health(cell, protocol, output)
            save_new(folder / "health.json", health)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            id=row["id"],
                            path=str(folder / "session.json"),
                            stop_policy_era=POLICY["version"],
                            status=cell["status"],
                            health=health,
                        )
                    )
                    + "\n"
                )
                stream.flush()
            print(json.dumps(dict(id=row["id"], status=cell["status"])), flush=True)
            if health["infrastructure_problem"] or cell["status"] not in {
                "complete",
                "failed_auth",
                "failed_capacity",
                "censored_receipt_timeout",
            }:
                failure = (
                    "driving_infrastructure_health"
                    if health["infrastructure_problem"]
                    else cell["status"]
                )
                break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_new(
            output / "batch.json",
            dict(
                protocol=study.PROTOCOL,
                cells=cells,
                failure=failure,
                checkpoint=addendum["original_checkpoint"],
                supervisor_policy=POLICY,
                supervisor_sha256=supervisor_hash,
                not_dispatched=addendum["assignments_remaining"][len(cells) :],
                deadline_reached=time.monotonic() >= deadline,
            ),
        )
        refs = [
            dict(
                id=cid,
                path=str(Path(addendum["previous_results"]) / cid / "session.json"),
                stop_policy_era="original_postdrive_stop_policy",
            )
            for cid in addendum["completed_ids"]
        ]
        refs.extend(
            dict(
                id=c["assignment"]["id"],
                path=str(output / c["assignment"]["id"] / "session.json"),
                stop_policy_era=POLICY["version"],
            )
            for c in cells
        )
        save_new(
            output / "union-manifest.json",
            dict(protocol_sha256=addendum["protocol_sha256"], cells=refs, completed=len(refs)),
        )
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    freezer = commands.add_parser("freeze")
    freezer.add_argument("protocol", type=Path)
    freezer.add_argument("old_results", type=Path)
    freezer.add_argument("output", type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("addendum", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze_addendum(args.protocol, args.old_results, args.output)
    else:
        run(args.addendum)


if __name__ == "__main__":
    main()
