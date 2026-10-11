import signal
import sys

import pytest
from drivingbench.sim.render_process import RenderProcess


def worker(tmp_path, body, timeout=2):
    script = tmp_path / "worker.py"
    script.write_text(body)
    return RenderProcess([sys.executable, "-u", script], tmp_path / "worker.log", timeout=timeout)


def test_process_ignores_logs_and_matches_frame(tmp_path):
    process = worker(
        tmp_path,
        """
import json, sys
print('renderer startup noise')
print('DRIVINGBENCH ' + json.dumps({'ready':True}))
for line in sys.stdin:
    request = json.loads(line)
    print('DRIVINGBENCH ' + json.dumps({'id':request['id'],'ok':True,'pose':request['pose']}))
""",
    )
    try:
        assert process.start()["ready"]
        assert process.request(pose=[1, 2, 3])["pose"] == [1, 2, 3]
    finally:
        process.close()
    assert process.process is None
    process.close()


def test_worker_uses_selected_interpreter_without_parent_python_environment(tmp_path, monkeypatch):
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, str(tmp_path / "wrong-environment"))
    monkeypatch.setenv("DRIVINGBENCH_WORKER_TEST", "preserved")
    process = worker(
        tmp_path,
        """
import json, os
from drivingbench.sim.actor_yield import validate_policy
import shapely
assert not any(key in os.environ for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'))
assert os.environ['DRIVINGBENCH_WORKER_TEST'] == 'preserved'
validate_policy({'kind': 'wait_for_stopped_car'})
print('DRIVINGBENCH ' + json.dumps({'ready': True}))
""",
    )
    try:
        assert process.start()["ready"]
    finally:
        process.close()


@pytest.mark.parametrize(
    "body", ["print('DRIVINGBENCH {}')", "raise SystemExit(2)", "import time; time.sleep(10)"]
)
def test_start_failure_cleans_up(tmp_path, body):
    process = worker(tmp_path, body, timeout=0.1)
    with pytest.raises((RuntimeError, TimeoutError)):
        process.start()
    assert process.process is None


def test_crash_does_not_return_old_frame(tmp_path):
    process = worker(
        tmp_path,
        """
import json, sys
print('DRIVINGBENCH {"ready":true}')
sys.stdin.readline()
raise SystemExit(3)
""",
    )
    process.start()
    try:
        with pytest.raises(RuntimeError, match="closed"):
            process.request(pose=[0, 0, 0])
    finally:
        process.close()


def test_concurrent_close_is_safe(tmp_path):
    import threading

    process = worker(
        tmp_path,
        """
import sys
print('DRIVINGBENCH {"ready":true}')
for line in sys.stdin:
    pass
""",
    )
    process.start()
    errors = []

    def close():
        try:
            process.close()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=close) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert process.process is None


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("error_type", [PermissionError, ProcessLookupError])
def test_signal_exit_race_reaps_owned_child(tmp_path, monkeypatch, sig, error_type):
    import os

    process = worker(
        tmp_path,
        "import sys\nprint('DRIVINGBENCH {\"ready\":true}')\nsys.stdin.readline()\nraise SystemExit(2)",
    )
    process.start()
    child = process.process

    def group_gone(pid, requested):
        assert pid == child.pid and requested == sig
        child.stdin.write("exit\n")
        child.stdin.flush()
        raise error_type("simulated exiting process-group race")

    monkeypatch.setattr(os, "killpg", group_gone)
    try:
        process._signal(child, sig)
        assert child.returncode == 2
    finally:
        process.close()
    assert process.process is None
    assert child.stdin.closed and child.stdout.closed


def test_signal_permission_failure_for_live_child_is_not_hidden(tmp_path, monkeypatch):
    import os

    process = worker(
        tmp_path,
        "import sys\nprint('DRIVINGBENCH {\"ready\":true}')\nfor line in sys.stdin: pass",
    )
    process.start()
    original = os.killpg

    def denied(*_):
        raise PermissionError("live process signal denied")

    monkeypatch.setattr(os, "killpg", denied)
    try:
        with pytest.raises(PermissionError, match="live process signal denied"):
            process.close()
        assert process.process.poll() is None
    finally:
        monkeypatch.setattr(os, "killpg", original)
        process.close()
    assert process.process is None


def test_start_failure_preserves_stream_error_during_permission_race(tmp_path, monkeypatch):
    import os

    process = worker(
        tmp_path,
        "import os,sys\nos.close(1)\nos.close(2)\nsys.stdin.readline()\nraise SystemExit(2)",
    )
    denied = []

    def exiting(*_):
        denied.append(True)
        process.process.stdin.write("exit\n")
        process.process.stdin.flush()
        raise PermissionError("exiting group")

    monkeypatch.setattr(os, "killpg", exiting)
    with pytest.raises(RuntimeError, match="Renderer stream closed"):
        process.start()
    assert denied
    assert process.process is None
