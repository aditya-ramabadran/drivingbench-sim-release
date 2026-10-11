"""Retrospective original fork coding; never starts or repeats a drive."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from drivingbench.replay import fork_analysis
from drivingbench.replay.fork_analysis import BELIEF_LABELS, CODER, RUBRIC
from drivingbench.sim.batch import save_new
from drivingbench.sim.producer import file_hash


def code_answer(answer, output, *, budget_s=180):
    """Reuse completed attempts, keep the first valid code, and allow one failed-call retry."""
    output = Path(output)
    identity = dict(answer=answer, coder=CODER, rubric=RUBRIC, budget_s=budget_s)
    source = output / "input.json"
    if source.exists():
        if json.loads(source.read_text()) != identity:
            raise ValueError("Cached belief input or original coder changed")
    else:
        save_new(source, identity)
    summary = output / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text())
    attempts = []
    for name in ("first", "retry") if answer.strip() else ():
        folder = output / name
        result_path = folder / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            if folder.exists():
                raise ValueError("Unfinished coder invocation; inspect it without replaying")
            save_new(folder / "dispatch.json", dict(budget_s=budget_s))
            prompt = RUBRIC.format(answer=answer)
            (folder / "prompt.txt").write_text(prompt)
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix="belief-coder-") as cwd:
                    reply = subprocess.run(
                        [*CODER, prompt], cwd=cwd, capture_output=True, timeout=budget_s
                    )
                raw, error, status = (
                    reply.stdout.decode(errors="replace"),
                    reply.stderr.decode(errors="replace"),
                    reply.returncode,
                )
            except subprocess.TimeoutExpired as exc:
                raw = (exc.stdout or b"").decode(errors="replace")
                error, status = "coder_timeout", None
            except OSError as exc:
                raw, error, status = "", str(exc), None
            (folder / "stdout.txt").write_text(raw)
            (folder / "stderr.txt").write_text(error)
            label = raw.lower().strip(" .`*\"'\n")
            result = dict(
                label=label if status == 0 and label in BELIEF_LABELS else None,
                raw=raw,
                returncode=status,
                wall_s=time.monotonic() - started,
            )
            save_new(result_path, result)
        attempts.append(dict(path=str(result_path), **result))
        if result["label"] in BELIEF_LABELS:
            break
    result = dict(
        label=attempts[-1]["label"] if attempts else None,
        missing=not attempts or attempts[-1]["label"] is None,
        attempts=attempts,
    )
    save_new(summary, result)
    return result


def code_results(results, output):
    """Code only preserved valid debriefs, serially; retain original run evidence."""
    results, output = Path(results), Path(output)
    manifest = json.loads((results / "manifest.json").read_text())
    if (results / "concurrent-execution.json").exists():
        closure = results / "summary.json"
        if (
            not closure.exists()
            or json.loads(closure.read_text()).get("all_waves_closed") is not True
        ):
            raise ValueError("Original retrospective coder must await all video waves closing")
    if manifest.get("protocol") != "trolley48" or len(manifest["assignments"]) != 48:
        raise ValueError("Expected the frozen48 trolley allocation")
    original = str(Path(fork_analysis.__file__).resolve())
    if manifest.get("source_hashes", {}).get(original) != file_hash(original):
        raise ValueError("Original frozen answer coder changed")
    reports = {}
    for row in manifest["assignments"]:
        path = results / row["id"] / "run.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        answer = report.get("perception", {}).get("belief_answer", "")
        if report.get("followups", {}).get("debrief", {}).get("missing", True):
            answer = ""
        folder = output / row["id"]
        reference = folder / "source.json"
        source = dict(path=str(path.resolve()), sha256=file_hash(path))
        if reference.exists():
            if json.loads(reference.read_text()) != source:
                raise ValueError("Preserved debrief changed")
        else:
            save_new(reference, source)
        reports[row["id"]] = code_answer(answer, folder)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code_results(args.results, args.output)


if __name__ == "__main__":
    main()
