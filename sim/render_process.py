"""Supervise a local JSONL renderer without mixing its logs and frame replies."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path

PREFIX = "DRIVINGBENCH "


class RenderProcess:
    def __init__(self, command, log: Path, *, timeout=60.0, cwd=None, new_session=True):
        self.command = list(map(str, command))
        self.log = Path(log)
        self.timeout = timeout
        self.cwd = cwd
        self.new_session = new_session
        self.process = None
        self.messages = queue.Queue()
        self.reader = None
        self.sequence = 0
        self._close_lock = threading.Lock()

    def start(self):
        if self.process is not None:
            raise RuntimeError("Renderer already started")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.messages = queue.Queue()
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=self.new_session,
            cwd=self.cwd,
            env={
                key: value
                for key, value in os.environ.items()
                if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
            },
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            message = self._receive()
            if message.get("ready") is not True:
                raise RuntimeError(f"Renderer did not become ready: {message}")
            return message
        except Exception:
            self.close()
            raise

    def _read(self):
        process = self.process
        with self.log.open("a") as log:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(PREFIX):
                        try:
                            self.messages.put(json.loads(line[len(PREFIX) :]))
                        except ValueError:
                            self.messages.put(RuntimeError("Malformed renderer reply"))
            finally:
                self.messages.put(RuntimeError("Renderer stream closed"))

    def _receive(self):
        try:
            message = self.messages.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError(f"Renderer timed out; see {self.log}") from error
        if isinstance(message, Exception):
            raise message
        if not isinstance(message, dict):
            raise RuntimeError("Renderer reply must be an object")
        return message

    def request(self, **payload):
        if self.process is None:
            raise RuntimeError("Renderer not started")
        self.sequence += 1
        request_id = str(self.sequence)
        self.process.stdin.write(json.dumps(payload | {"id": request_id}, allow_nan=False) + "\n")
        self.process.stdin.flush()
        result = self._receive()
        if result.get("id") != request_id:
            raise RuntimeError("Renderer reply does not match requested frame")
        if result.get("ok") is not True:
            raise RuntimeError(result.get("error", "Renderer failed"))
        return result

    def _signal(self, process, sig):
        try:
            if self.new_session:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except (ProcessLookupError, PermissionError) as error:
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                raise error

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            self._signal(process, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._signal(process, signal.SIGKILL)
                process.wait(timeout=3)
        if self.reader:
            self.reader.join(timeout=3)
        process.stdin.close()
        process.stdout.close()
        self.process = None
