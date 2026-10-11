import json
from types import SimpleNamespace

import pytest
from drivingbench.replay.fork_analysis import CODER, RUBRIC
from drivingbench.sim import video_belief as belief


def test_original_coder_one_failed_retry_then_cached_first_valid(tmp_path, monkeypatch):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=1 if len(calls) == 1 else 0,
            stdout=b"not a code" if len(calls) == 1 else b"doubted",
            stderr=b"",
        )

    monkeypatch.setattr(belief.subprocess, "run", invoke)
    result = belief.code_answer("I doubted it.", tmp_path / "q")
    assert result["label"] == "doubted" and len(calls) == 2
    assert calls == [[*CODER, RUBRIC.format(answer="I doubted it.")]] * 2
    assert belief.code_answer("I doubted it.", tmp_path / "q") == result and len(calls) == 2
    with pytest.raises(ValueError, match="changed"):
        belief.code_answer("Different answer", tmp_path / "q")


def test_coder_does_not_recode_valid_first_attempt_or_invent_missing_answer(tmp_path, monkeypatch):
    calls = []

    def invoke(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=b"not_real", stderr=b"")

    monkeypatch.setattr(belief.subprocess, "run", invoke)
    result = belief.code_answer("No.", tmp_path / "valid")
    assert result["label"] == "not_real" and len(calls) == 1
    (tmp_path / "valid/summary.json").unlink()
    assert belief.code_answer("No.", tmp_path / "valid") == result and len(calls) == 1
    assert belief.code_answer("", tmp_path / "empty")["missing"] and len(calls) == 1


def test_coder_failure_is_missing_after_two_and_ambiguous_invocation_not_replayed(
    tmp_path, monkeypatch
):
    calls = []

    def invoke(*args, **kwargs):
        calls.append(args)
        raise OSError("unavailable")

    monkeypatch.setattr(belief.subprocess, "run", invoke)
    result = belief.code_answer("Maybe.", tmp_path / "failed")
    assert result["missing"] and len(calls) == 2
    assert belief.code_answer("Maybe.", tmp_path / "failed") == result and len(calls) == 2
    q = tmp_path / "ambiguous"
    q.mkdir()
    (q / "first").mkdir()
    (q / "input.json").write_text(
        json.dumps(dict(answer="A", coder=CODER, rubric=RUBRIC, budget_s=180))
    )
    with pytest.raises(ValueError, match="Unfinished"):
        belief.code_answer("A", q)
    assert len(calls) == 2
