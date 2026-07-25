"""The fleet's "which account should be active" record.

One small file, ``<backup_root>/active_intent.json``, holds the most recent
switch this device knows about — its own or a peer's. Intents are keyed by
account identity (email + organizationUuid), never slot number: slots are
per-device. Ordering is last-writer-wins on the ``(ts, originDeviceId)``
tuple, the same strictly-fresher rule usage gossip uses for measurements,
so re-encountering an already-adopted intent is always a no-op and no
sync cycle can amplify a switch.

Kept out of sequence.json deliberately: that file has no schemaVersion and
is rewritten by many code paths; this one has a single writer path guarded
by its own lock.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json

INTENT_FILENAME = "active_intent.json"
INTENT_LOCK_FILENAME = ".active_intent.lock"
INTENT_SCHEMA_VERSION = 1

_ORIGIN_KINDS = ("manual", "auto")

_logger = logging.getLogger("claude-swap")


def intent_path(backup_root: Path) -> Path:
    return backup_root / INTENT_FILENAME


def _lock(backup_root: Path) -> FileLock:
    return FileLock(backup_root / INTENT_LOCK_FILENAME)


def load_intent(backup_root: Path) -> dict | None:
    """The device's current intent record; missing/corrupt file → None."""
    try:
        raw = json.loads(intent_path(backup_root).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        _logger.warning("Could not read %s (%s); ignoring", INTENT_FILENAME, e)
        return None
    if not isinstance(raw, dict) or raw.get("schemaVersion") != INTENT_SCHEMA_VERSION:
        return None
    intent = raw.get("intent")
    return validate_intent_record(intent)


def validate_intent_record(intent: object) -> dict | None:
    """Normalize one intent record; anything malformed → None.

    Accepts records from disk and from peers alike, so every field is
    treated as untrusted.
    """
    if not isinstance(intent, dict):
        return None
    email = intent.get("email")
    ts = intent.get("ts")
    device = intent.get("originDeviceId")
    kind = intent.get("originKind")
    if not isinstance(email, str) or not email:
        return None
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    if not isinstance(device, str) or not device:
        return None
    if kind not in _ORIGIN_KINDS:
        return None
    org = intent.get("organizationUuid")
    adopted = intent.get("adoptedFrom")
    return {
        "email": email,
        "organizationUuid": org if isinstance(org, str) else "",
        "ts": float(ts),
        "originDeviceId": device,
        "originKind": kind,
        "adoptedFrom": adopted if isinstance(adopted, str) else None,
    }


def validate_intent(payload: object) -> dict | None:
    """Validate a gossip payload ({schemaVersion, intent}); bad → None."""
    if not isinstance(payload, dict):
        return None
    if payload.get("schemaVersion") != INTENT_SCHEMA_VERSION:
        return None
    return validate_intent_record(payload.get("intent"))


def intent_key(intent: dict | None) -> tuple[float, str]:
    """LWW ordering key. Malformed/absent sorts before everything."""
    if not isinstance(intent, dict):
        return (float("-inf"), "")
    ts = intent.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return (float("-inf"), "")
    device = intent.get("originDeviceId")
    return (float(ts), device if isinstance(device, str) else "")


def is_newer(candidate: dict, baseline: dict | None) -> bool:
    """Strictly-greater LWW: ties lose, so re-applying is always a no-op."""
    if baseline is None:
        return True
    return intent_key(candidate) > intent_key(baseline)


def _write(backup_root: Path, intent: dict) -> None:
    atomic_write_json(
        intent_path(backup_root),
        {"schemaVersion": INTENT_SCHEMA_VERSION, "intent": intent},
    )


def record_local_intent(
    backup_root: Path,
    *,
    email: str,
    org_uuid: str,
    device_id: str,
    kind: str,
) -> dict:
    """Mint a new intent originating on this device.

    The timestamp is nudged past the previous record's so a
    backwards-stepped clock still produces an intent that beats this
    device's own prior one.
    """
    if kind not in _ORIGIN_KINDS:
        raise ValueError(f"unknown intent origin kind {kind!r}")
    with _lock(backup_root):
        prev = load_intent(backup_root)
        ts = time.time()
        if prev is not None:
            ts = max(ts, float(prev["ts"]) + 0.001)
        intent = {
            "email": email,
            "organizationUuid": org_uuid or "",
            "ts": ts,
            "originDeviceId": device_id,
            "originKind": kind,
            "adoptedFrom": None,
        }
        _write(backup_root, intent)
    return intent


def adopt_intent(backup_root: Path, intent: dict, *, source: str) -> bool:
    """Store a peer's intent verbatim (same ts/origin — never re-originated).

    Re-checks freshness under the lock so a newer record that landed between
    the caller's check and this write is never clobbered. Returns False when
    the intent is no longer the newest.
    """
    normalized = validate_intent_record(intent)
    if normalized is None:
        return False
    with _lock(backup_root):
        current = load_intent(backup_root)
        if not is_newer(normalized, current):
            return False
        normalized["adoptedFrom"] = source or None
        _write(backup_root, normalized)
    return True
