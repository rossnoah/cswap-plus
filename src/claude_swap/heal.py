"""Reactive credential healing across the sync fleet.

When a device concludes a refresh-token lineage is dead (``invalid_grant``
— one strike is definitive), any peer that holds a *different* generation
of that account can donate it: the dead copy is beyond saving, so adopting
any fingerprint-differing generation is strictly better than waiting for a
re-login. That criterion is deliberately NOT the strict expiresAt ordering
used by import freshening — the dead local copy can carry a later expiry
than the live successor (the last successful refresh stamped it before the
lineage was lost), so "fresher" would reject the only working credential
in the fleet. Among candidates, the highest expiresAt wins (most recently
refreshed → most likely the head of the lineage).

Adopted credentials are never verification-refreshed: a refresh grant
consumes a generation and instantly stales the donor's copy — the exact
failure class this module exists to repair. The normal poll/freshen
machinery proves the adoption within minutes; if it also dies, its
fingerprint lands in the dead-fingerprint ledger and is never re-adopted,
and per-identity exponential backoff keeps a fleet with no working copy
from SSH-storming.

State lives in ``<backup_root>/cache/heal.json`` under its own lock, with
the usage-store claim discipline: lock → check backoff + stamp the attempt
→ unlock → network → lock → record outcome → unlock. No network under
locks, ever.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from claude_swap import oauth
from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json, load_sync_section_settings
from claude_swap.sync import _run_remote, load_sync_config

HEAL_SCHEMA_VERSION = 1
HEAL_BACKOFF_BASE_S = 300.0
HEAL_BACKOFF_CAP_S = 14400.0
DEAD_FINGERPRINT_TTL_S = 30 * 86400

_logger = logging.getLogger("claude-swap")


@dataclass(frozen=True)
class HealOutcome:
    """One heal attempt's result.

    ``healed-live`` means the adopted generation was also force-activated
    as the live login; ``healed`` with a live session running notes the
    skipped activation in ``detail``.
    """

    status: str  # healed | healed-live | no-peers | no-donor | cooldown
    #             | disabled | raced | skipped-api-key | error
    host: str | None = None
    fingerprint: str | None = None
    detail: str = ""


def heal_state_path(backup_root: Path) -> Path:
    return backup_root / "cache" / "heal.json"


def _lock(backup_root: Path) -> FileLock:
    return FileLock(backup_root / "cache" / ".heal.lock")


def _identity_key(email: str, org_uuid: str) -> str:
    return f"{email}|{org_uuid or ''}"


def _read_state(backup_root: Path) -> dict:
    try:
        raw = json.loads(heal_state_path(backup_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _mutate_state(backup_root: Path, mutator: Callable[[dict], None]) -> dict:
    with _lock(backup_root):
        state = _read_state(backup_root)
        state["schemaVersion"] = HEAL_SCHEMA_VERSION
        mutator(state)
        _prune_dead_fingerprints(state)
        atomic_write_json(heal_state_path(backup_root), state)
        return state


def _prune_dead_fingerprints(state: dict) -> None:
    cutoff = time.time() - DEAD_FINGERPRINT_TTL_S
    for entry in (state.get("identities") or {}).values():
        fps = entry.get("deadFingerprints")
        if isinstance(fps, dict):
            for fp in [f for f, at in fps.items() if not isinstance(at, (int, float)) or at < cutoff]:
                del fps[fp]


def _entry(state: dict, key: str) -> dict:
    return (state.get("identities") or {}).get(key) or {}


def _backoff_s(consecutive_failures: int) -> float:
    if consecutive_failures <= 0:
        return 0.0
    return min(
        HEAL_BACKOFF_CAP_S,
        HEAL_BACKOFF_BASE_S * (2 ** (consecutive_failures - 1)),
    )


def note_dead_fingerprint(
    backup_root: Path, email: str, org_uuid: str, fingerprint: str | None
) -> None:
    """Ledger a fingerprint that just proved dead, so it is never re-adopted."""
    if not fingerprint:
        return
    key = _identity_key(email, org_uuid)

    def add(state: dict) -> None:
        entry = state.setdefault("identities", {}).setdefault(key, {})
        entry.setdefault("deadFingerprints", {})[fingerprint] = time.time()

    _mutate_state(backup_root, add)


def note_live_fingerprint(
    backup_root: Path, email: str, org_uuid: str, fingerprint: str | None
) -> None:
    """Drop a fingerprint that just proved alive from the dead ledger.

    A successful fetch outranks any past ``invalid_grant`` verdict (a strike
    mis-attributed during a racing freshen, a server hiccup): the lineage is
    demonstrably alive, so un-ledger it and end the identity's heal backoff.
    This is what makes a poisoned ledger self-correct instead of refusing
    the fleet's working generation for the 30-day TTL. Lock-free fast path —
    nearly every poll succeeds against an empty ledger.
    """
    if not fingerprint:
        return
    key = _identity_key(email, org_uuid)
    entry = _entry(_read_state(backup_root), key)
    fps = entry.get("deadFingerprints")
    ledgered = isinstance(fps, dict) and fingerprint in fps
    if not ledgered and not entry.get("consecutiveFailures"):
        return

    def drop(state: dict) -> None:
        e = state.setdefault("identities", {}).setdefault(key, {})
        d = e.get("deadFingerprints")
        if isinstance(d, dict):
            d.pop(fingerprint, None)
        e["consecutiveFailures"] = 0
        e["lastOutcome"] = "proven-live"

    _mutate_state(backup_root, drop)
    if ledgered:
        _logger.info(
            "heal %s: fingerprint proved alive, removed from dead ledger", email
        )


def _quarantine_fingerprint(backup_root: Path, slot: str) -> str | None:
    """The dead fingerprint autoswitch recorded for a slot, if any."""
    try:
        raw = json.loads(
            (backup_root / "autoswitch_state.json").read_text(encoding="utf-8")
        )
        entry = (raw.get("quarantine") or {}).get(slot) or {}
        fp = entry.get("refreshTokenFingerprint")
        return fp if isinstance(fp, str) and fp else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def heal_from_peers(
    switcher,
    email: str,
    org_uuid: str,
    *,
    hosts: tuple[str, ...] | None = None,
    activate_live: bool = True,
    emit: Callable[[HealOutcome], None] | None = None,
    quiet: bool = True,
) -> HealOutcome:
    """Pull a working generation of one dead identity from the peers."""

    def done(outcome: HealOutcome) -> HealOutcome:
        if emit is not None:
            try:
                emit(outcome)
            except Exception:
                pass
        if not quiet:
            line = f"heal {email}: {outcome.status}"
            if outcome.host:
                line += f" (from {outcome.host})"
            if outcome.detail:
                line += f" — {outcome.detail}"
            print(line)
        _logger.info(
            "heal %s: %s%s", email, outcome.status,
            f" from {outcome.host}" if outcome.host else "",
        )
        return outcome

    backup_root = switcher.backup_dir
    org_uuid = org_uuid or ""
    key = _identity_key(email, org_uuid)

    try:
        if not load_sync_section_settings(backup_root).heal_on_death:
            return done(HealOutcome("disabled", detail="sync.healOnDeath=false"))

        data = switcher._get_sequence_data() or {}
        slot = switcher._find_account_slot(data, email, org_uuid)
        if slot is None:
            return done(HealOutcome("error", detail=f"{email} is not managed here"))
        if switcher.account_kind_for(slot) == "api_key":
            return done(HealOutcome("skipped-api-key", detail="API keys never rotate"))

        if hosts is None:
            hosts = load_sync_config(backup_root, create=False).peers
        if not hosts:
            # The ubuntu path: no outbound SSH, healed by inbound pushes.
            # Deliberately records nothing — a no-peer device must never
            # accrue backoff state.
            return done(HealOutcome("no-peers", detail="no sync peers configured"))

        # Claim: check backoff and stamp the attempt in one lock window.
        now = time.time()
        claimed = {"ok": False}

        def claim(state: dict) -> None:
            entry = state.setdefault("identities", {}).setdefault(key, {})
            failures = int(entry.get("consecutiveFailures") or 0)
            last = entry.get("lastAttemptAt")
            if isinstance(last, (int, float)) and now < last + _backoff_s(failures):
                return
            entry["lastAttemptAt"] = now
            claimed["ok"] = True

        state = _mutate_state(backup_root, claim)
        if not claimed["ok"]:
            return done(HealOutcome("cooldown", detail="heal attempted recently"))

        # The known-dead set: the local copy, autoswitch's quarantine record,
        # and every fingerprint the ledger has seen die.
        local_creds = switcher.read_account_credentials(slot, email) or ""
        local_fp = oauth.credential_fingerprint(local_creds)
        dead_fps = {fp for fp in (local_fp,) if fp}
        quarantine_fp = _quarantine_fingerprint(backup_root, slot)
        if quarantine_fp:
            dead_fps.add(quarantine_fp)
        ledger = _entry(state, key).get("deadFingerprints")
        if isinstance(ledger, dict):
            dead_fps.update(ledger)

        # Network, no locks held: ask each peer for its copy.
        identity_payload = json.dumps(
            {"schemaVersion": 1, "email": email, "organizationUuid": org_uuid}
        ).encode()
        best: tuple[int, str, str, str] | None = None  # (expiry, host, creds, config)
        for host in hosts:
            try:
                proc = _run_remote(host, "sync emit-credential -", identity_payload)
            except Exception as exc:
                _logger.info("heal: %s unreachable (%s)", host, exc)
                continue
            if proc.returncode != 0:
                # rc 3: peer doesn't hold this identity (or it's an API key
                # there); anything else: old cswap / transport — next host.
                continue
            candidate = _extract_candidate(proc.stdout, email, org_uuid)
            if candidate is None:
                continue
            creds_text, config_text = candidate
            fp = oauth.credential_fingerprint(creds_text)
            if not fp or fp in dead_fps:
                continue
            expiry = oauth.credential_expires_at(creds_text) or -1
            if best is None or expiry > best[0]:
                best = (expiry, host, creds_text, config_text)

        if best is None:
            def fail(state: dict) -> None:
                entry = state.setdefault("identities", {}).setdefault(key, {})
                entry["consecutiveFailures"] = int(
                    entry.get("consecutiveFailures") or 0
                ) + 1
                entry["lastOutcome"] = "no-donor"

            _mutate_state(backup_root, fail)
            return done(HealOutcome(
                "no-donor", detail="no peer holds a different generation"
            ))

        _expiry, host, creds_text, config_text = best
        new_fp = oauth.credential_fingerprint(creds_text)

        # Adopt under the backup lock, re-checking that the local credential
        # is still the dead one — a concurrent re-login or heal must win.
        with FileLock(switcher.lock_file):
            current = switcher.read_account_credentials(slot, email) or ""
            current_fp = oauth.credential_fingerprint(current)
            if current_fp != local_fp:
                return done(HealOutcome(
                    "raced", detail="credential changed during the heal"
                ))
            switcher._write_account_credentials(slot, email, creds_text)
            if config_text:
                switcher._write_account_config(slot, email, config_text)
        switcher._usage_store.clear_dead_token(
            [slot], {slot: (email, org_uuid)}
        )

        def succeed(state: dict) -> None:
            entry = state.setdefault("identities", {}).setdefault(key, {})
            entry["consecutiveFailures"] = 0
            entry["lastOutcome"] = "healed"
            entry["lastAdoptedFingerprint"] = new_fp
            entry["lastAdoptedFrom"] = host

        _mutate_state(backup_root, succeed)

        # Live-login repair: when this identity IS the live login, land the
        # working generation in the live store too, before the stale copy
        # dies in Claude Code's hands. A live `cswap run` session owning the
        # slot skips only this activation step.
        if activate_live and switcher._get_current_account() == (email, org_uuid):
            if switcher.live_session_pids_for(slot, email):
                return done(HealOutcome(
                    "healed", host=host, fingerprint=new_fp,
                    detail="backup healed; live session active, not activated",
                ))
            try:
                switcher.switch_to(
                    slot, json_output=True, force=True, origin="remote-heal"
                )
            except Exception as exc:
                return done(HealOutcome(
                    "healed", host=host, fingerprint=new_fp,
                    detail=f"backup healed; live activation failed: {exc}",
                ))
            return done(HealOutcome("healed-live", host=host, fingerprint=new_fp))

        return done(HealOutcome("healed", host=host, fingerprint=new_fp))
    except Exception as exc:  # a heal must never take its caller down
        _logger.warning("heal %s failed: %s", email, exc)
        return HealOutcome("error", detail=str(exc))


def _extract_candidate(
    stdout: bytes, email: str, org_uuid: str
) -> tuple[str, str] | None:
    """Pull (creds_text, config_text) for one identity out of an envelope."""
    try:
        envelope = json.loads(stdout.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(envelope, dict):
        return None
    for raw in envelope.get("accounts") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("email") != email:
            continue
        if (raw.get("organizationUuid") or "") != org_uuid:
            continue
        creds = raw.get("credentials")
        if not isinstance(creds, dict):  # API key or garbage — never adopt
            return None
        config = raw.get("config")
        return (
            json.dumps(creds),
            json.dumps(config, indent=2) if isinstance(config, dict) else "",
        )
    return None


def dead_identities(switcher) -> list[tuple[str, str, str]]:
    """(slot, email, org) for every managed identity currently judged dead."""
    data = switcher._get_sequence_data() or {}
    accounts = data.get("accounts") or {}
    identities = {
        num: (acc.get("email") or "", acc.get("organizationUuid") or "")
        for num, acc in accounts.items()
        if isinstance(acc, dict) and acc.get("email")
    }
    if not identities:
        return []
    entries = switcher._usage_store.entries(identities)
    quarantined: set[str] = set()
    try:
        raw = json.loads(
            (switcher.backup_dir / "autoswitch_state.json").read_text(
                encoding="utf-8"
            )
        )
        for num, entry in (raw.get("quarantine") or {}).items():
            if isinstance(entry, dict) and entry.get("reason") == "invalid_grant":
                quarantined.add(num)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return [
        (num, email, org)
        for num, (email, org) in identities.items()
        if (num in entries and entries[num].token_dead()) or num in quarantined
    ]


def heal_all_dead(switcher, *, quiet: bool = True) -> list[HealOutcome]:
    """Heal every identity this device currently judges dead."""
    outcomes = []
    for _slot, email, org in dead_identities(switcher):
        outcomes.append(
            heal_from_peers(switcher, email, org, quiet=quiet)
        )
    return outcomes


def on_death_detected(
    switcher, items: list[tuple[str, str, str, str | None]]
) -> None:
    """Best-effort death hook for interactive surfaces: ledger the dying
    fingerprints, then hand off to a detached heal so no interactive path
    ever blocks on SSH. ``items`` is [(slot, email, org_uuid, fingerprint)],
    where ``fingerprint`` identifies the credential that actually failed,
    captured at fetch time. The slot is deliberately NOT re-read here: a
    sync freshen racing this hook can have already replaced the dead bytes,
    and fingerprinting the replacement would ledger the fleet's only working
    generation as dead."""
    try:
        if not load_sync_section_settings(switcher.backup_dir).heal_on_death:
            return
        for _slot, email, org, fingerprint in items:
            note_dead_fingerprint(switcher.backup_dir, email, org, fingerprint)
        spawn_background_heal(switcher)
    except Exception as exc:
        _logger.info("death hook skipped: %s", exc)


def spawn_background_heal(switcher) -> bool:
    """Detach a ``cswap sync heal --quiet`` process; True when spawned."""
    try:
        if not load_sync_config(switcher.backup_dir, create=False).peers:
            return False
        subprocess.Popen(
            [sys.executable, "-m", "claude_swap", "sync", "heal", "--quiet"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:
        _logger.info("background heal not spawned: %s", exc)
        return False
