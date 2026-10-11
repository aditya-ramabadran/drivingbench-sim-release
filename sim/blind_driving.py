"""Client receipt gaps at physical events, with bounded clock registration."""

from datetime import datetime
from math import isfinite
from statistics import median


def clock_bounds(deliveries):
    """Bound UTC minus monotonic from wall samples inside each observe invocation."""
    intervals = []
    for row in deliveries:
        stamp = row.get("original_wall_timestamp")
        if stamp is None:
            continue
        utc = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        start, end = row["arrived_at"], row["delivered_at"]
        if not all(isfinite(v) for v in (utc, start, end)) or end < start:
            return dict(status="invalid_anchor", offset_s=None)
        intervals.append((utc - end, utc - start))
    if not intervals:
        return dict(status="no_anchor", offset_s=None)
    lower, upper = max(i[0] for i in intervals), min(i[1] for i in intervals)
    return dict(
        status="bounded" if lower <= upper else "inconsistent_clock",
        offset_s=[lower, upper] if lower <= upper else None,
        anchors=len(intervals),
    )


def exposure_at(monotonic_s, observations, clock):
    """Keep time since receipt distinct from capture age; paired frames count once."""
    bounds = clock.get("offset_s")
    if bounds is None or monotonic_s is None:
        return dict(status="unknown_clock", receipt_gap_s=None, capture_age_s=None)
    event = [monotonic_s + b for b in bounds]
    before = [
        (i, o)
        for i, o in enumerate(observations)
        if o.get("received_utc_s") is not None and o["received_utc_s"] <= event[0]
    ]
    index, last = max(before, key=lambda pair: pair[1]["received_utc_s"], default=(-1, None))
    ambiguous = any(
        o.get("received_utc_s") is not None and event[0] < o["received_utc_s"] <= event[1]
        for o in observations
    )
    unknown_later = any(o.get("received_utc_s") is None for o in observations[index + 1 :])
    result = dict(
        status="ambiguous_receipt" if ambiguous or unknown_later else "measured",
        event_utc_s=event,
        last_definite_call_id=last["call_id"] if last else None,
        receipt_gap_s=[v - last["received_utc_s"] for v in event] if last else None,
        capture_age_s={
            f["camera"]: monotonic_s - f["captured_monotonic_s"]
            for f in last.get("frames", [])
            if f.get("captured_monotonic_s") is not None
        }
        if last
        else None,
    )
    if last is None and not ambiguous and not unknown_later:
        result["status"] = "no_prior_receipt"
    if ambiguous or unknown_later:
        result["receipt_gap_s"] = None
        result["capture_age_s"] = None
    return result


def receipt_cadence(observations, commands, clock):
    """Compare authored durations with actual receipt intervals and active windows."""
    known = [o for o in observations if o.get("received_utc_s") is not None]
    known.sort(key=lambda o: o["received_utc_s"])
    gaps = [b["received_utc_s"] - a["received_utc_s"] for a, b in zip(known, known[1:])]
    identities, fresh, repeated = set(), [], []
    for row in known:
        keys = {
            (f["camera"], f.get("captured_monotonic_s", f["frame_id"]))
            for f in row.get("frames", [])
        }
        (fresh if keys - identities else repeated).append(row)
        identities.update(keys)
    bounds = clock.get("offset_s")
    windows = []
    for command in commands:
        start, end = command.get("accepted_monotonic_s"), command.get("end_monotonic_s")
        interval = None
        if bounds is not None and start is not None and end is not None:
            earliest_end = min(end, (command.get("early_clearance_interval_s") or [end])[0])
            interval = [start + bounds[1], earliest_end + bounds[0]]
        inside = (
            [o["call_id"] for o in known if interval[0] <= o["received_utc_s"] < interval[1]]
            if interval
            else None
        )
        new = (
            [o["call_id"] for o in fresh if interval[0] <= o["received_utc_s"] < interval[1]]
            if interval
            else None
        )
        uncertain = (
            [
                o["call_id"]
                for o in known
                if start + bounds[0] <= o["received_utc_s"] < end + bounds[1]
                and o["call_id"] not in inside
            ]
            if interval
            else None
        )
        windows.append(
            dict(
                **command,
                received_while_active=inside,
                fresh_while_active=new,
                ambiguous_boundary_receipts=uncertain,
                ended_without_fresh_receipt=False
                if new
                else True
                if new is not None and not uncertain and len(known) == len(observations)
                else None,
            )
        )
    durations = [c["requested_duration_s"] for c in commands if "requested_duration_s" in c]
    return dict(
        median_requested_duration_s=median(durations) if durations else None,
        observations=len(observations),
        timestamped_observations=len(known),
        receipt_intervals_s=gaps,
        median_receipt_interval_s=median(gaps) if gaps else None,
        fresh_observations=len(fresh),
        repeated_observations=len(repeated),
        commands=windows,
    )
