"""Read openpilot rlogs with the vendored cereal schema (no openpilot runtime needed)."""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CEREAL = REPO / "vendor/openpilot/openpilot/cereal/log.capnp"
CAR = REPO / "vendor/openpilot/opendbc_repo/opendbc/car"
OPENPILOT = (REPO / "vendor/openpilot", REPO / "vendor/openpilot/opendbc_repo")


def use_openpilot():
    """Make the vendored `openpilot` and `opendbc` packages importable."""
    for path in map(str, reversed(OPENPILOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


@cache
def schema():
    import capnp

    capnp.remove_import_hook()
    return capnp.load(str(CEREAL), imports=[str(CAR)])


def events(path: Path, types: set[str] | None = None):
    """(logMonoTime ns, type, message) for each event, optionally only the given types."""
    import io

    import zstandard

    raw = Path(path).read_bytes()
    if Path(path).suffix == ".zst":
        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
    for event in schema().Event.read_multiple_bytes(raw):
        kind = event.which()
        if types is None or kind in types:
            yield event.logMonoTime, kind, getattr(event, kind)


def car_params(raw: bytes):
    """The logged CarParams, usable wherever openpilot expects CP."""
    import capnp

    schema()
    with capnp.load(str(CAR / "car.capnp"), imports=[str(CAR)]).CarParams.from_bytes(raw) as cp:
        return cp.as_builder().as_reader()
