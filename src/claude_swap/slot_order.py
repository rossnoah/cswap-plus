"""The fleet's "which account sits in which slot" ordering record.

Slot layout (the order ``cswap list`` shows) is per-device state that the
user edits with add/remove/swap/move. To keep the whole fleet showing the
same list, each explicit layout change stamps this device as the latest
layout author in ``<backup_root>/slot_order.json``; the export envelope
carries the stamp alongside the accounts' slot numbers, and import
rearranges local slots to match whenever the incoming stamp is strictly
newer (same LWW ordering as the active-intent record — ties lose, adopted
stamps travel verbatim, so re-encounters are no-ops and nothing echoes).

Only the *stamp* lives here; the layout itself is sequence.json — the
envelope's per-account ``number`` fields are the layout in transit.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json

ORDER_FILENAME = "slot_order.json"
ORDER_LOCK_FILENAME = ".slot_order.lock"
ORDER_SCHEMA_VERSION = 1

_logger = logging.getLogger("claude-swap")


def order_path(backup_root: Path) -> Path:
    return backup_root / ORDER_FILENAME


def _lock(backup_root: Path) -> FileLock:
    return FileLock(backup_root / ORDER_LOCK_FILENAME)


def validate_record(record: object) -> dict | None:
    """Normalize one order stamp; anything malformed → None."""
    if not isinstance(record, dict):
        return None
    ts = record.get("ts")
    device = record.get("originDeviceId")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    if not isinstance(device, str) or not device:
        return None
    adopted = record.get("adoptedFrom")
    return {
        "ts": float(ts),
        "originDeviceId": device,
        "adoptedFrom": adopted if isinstance(adopted, str) else None,
    }


def load_order(backup_root: Path) -> dict | None:
    try:
        raw = json.loads(order_path(backup_root).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        _logger.warning("Could not read %s (%s); ignoring", ORDER_FILENAME, e)
        return None
    if not isinstance(raw, dict) or raw.get("schemaVersion") != ORDER_SCHEMA_VERSION:
        return None
    return validate_record(raw.get("order"))


def _key(record: dict | None) -> tuple[float, str]:
    if not isinstance(record, dict):
        return (float("-inf"), "")
    ts = record.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return (float("-inf"), "")
    device = record.get("originDeviceId")
    return (float(ts), device if isinstance(device, str) else "")


def is_newer(candidate: dict, baseline: dict | None) -> bool:
    """Strictly-greater LWW on (ts, originDeviceId); ties lose."""
    if baseline is None:
        return True
    return _key(candidate) > _key(baseline)


def _write(backup_root: Path, record: dict) -> None:
    atomic_write_json(
        order_path(backup_root),
        {"schemaVersion": ORDER_SCHEMA_VERSION, "order": record},
    )


def record_local_order(backup_root: Path, device_id: str) -> dict:
    """Stamp this device as the latest layout author (add/remove/swap/move).

    Monotonic-guarded like the active-intent record: a backwards-stepped
    clock still beats this device's own previous stamp.
    """
    with _lock(backup_root):
        prev = load_order(backup_root)
        ts = time.time()
        if prev is not None:
            ts = max(ts, float(prev["ts"]) + 0.001)
        record = {"ts": ts, "originDeviceId": device_id, "adoptedFrom": None}
        _write(backup_root, record)
    return record


def adopt_record(backup_root: Path, record: dict, *, source: str) -> bool:
    """Store a peer's stamp verbatim after applying its layout; False when
    a newer stamp landed in the meantime (TOCTOU re-check under the lock)."""
    normalized = validate_record(record)
    if normalized is None:
        return False
    with _lock(backup_root):
        current = load_order(backup_root)
        if not is_newer(normalized, current):
            return False
        normalized["adoptedFrom"] = source or None
        _write(backup_root, normalized)
    return True
