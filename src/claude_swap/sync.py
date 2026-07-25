"""Cross-device account sync over SSH (``cswap sync``).

Peers are plain SSH destinations (``mm``, ``noah@ubuntu``) saved in
``<backup_root>/sync.json``. A sync pulls each peer's accounts and pushes
ours, reusing the export/import envelope end to end: the remote side runs
``cswap export -`` / ``cswap import -``, so both directions inherit
import's validation and conflict rules (heal dead tokens, skip healthy
accounts unless ``--force``). Credentials only ever transit the SSH
channel; temp files stay 0600 inside the backup root and are deleted.

The saved peer list doubles as the device fleet for poll-budget sharing
(see ``poll_policy``): ``device_id`` gives each device a stable identity.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from claude_swap import active_intent
from claude_swap.exceptions import SyncError
from claude_swap.printer import accent, dimmed, warning
from claude_swap.settings import atomic_write_json
from claude_swap.transfer import export_accounts, import_accounts

SYNC_FILENAME = "sync.json"
SCHEMA_VERSION = 1

# Non-interactive SSH commonly skips the profile that puts ~/.local/bin
# (the uv tool dir) on PATH, so spell it out on the remote side.
_REMOTE_CSWAP = 'PATH="$HOME/.local/bin:$PATH" cswap'
_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
_REMOTE_TIMEOUT_S = 120

# SSH destination: alias, host, user@host — never something option-shaped.
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")


@dataclass(frozen=True)
class SyncConfig:
    device_id: str
    peers: tuple[str, ...] = ()


def sync_config_path(backup_root: Path) -> Path:
    return backup_root / SYNC_FILENAME


def load_sync_config(backup_root: Path, *, create: bool = True) -> SyncConfig:
    """Read sync.json, minting (and persisting) a device id on first use.

    ``create=False`` is the read-only form for surfaces that only consult the
    config (poll-budget sharing): no sync.json appears in the backup roots of
    users who never touch ``cswap sync``; the device id is ``""`` until then.
    """
    path = sync_config_path(backup_root)
    raw: dict = {}
    try:
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            raw = loaded
    except (OSError, json.JSONDecodeError):
        pass
    device_id = raw.get("deviceId")
    peers = tuple(
        p for p in raw.get("peers", []) if isinstance(p, str) and _HOST_RE.match(p)
    )
    if not isinstance(device_id, str) or not device_id:
        if not create:
            return SyncConfig(device_id="", peers=peers)
        device_id = uuid.uuid4().hex
        _save(backup_root, SyncConfig(device_id=device_id, peers=peers))
    return SyncConfig(device_id=device_id, peers=peers)


def _save(backup_root: Path, config: SyncConfig) -> None:
    atomic_write_json(
        sync_config_path(backup_root),
        {
            "schemaVersion": SCHEMA_VERSION,
            "deviceId": config.device_id,
            "peers": list(config.peers),
        },
    )


def validate_host(host: str) -> str:
    if not _HOST_RE.match(host):
        raise SyncError(
            f"invalid SSH destination '{host}' (use an ssh alias, host, or user@host)"
        )
    return host


def add_peer(backup_root: Path, host: str) -> bool:
    """Add a peer; returns False if it was already saved."""
    validate_host(host)
    config = load_sync_config(backup_root)
    if host in config.peers:
        return False
    _save(backup_root, SyncConfig(config.device_id, config.peers + (host,)))
    return True


def remove_peer(backup_root: Path, host: str) -> bool:
    config = load_sync_config(backup_root)
    if host not in config.peers:
        return False
    peers = tuple(p for p in config.peers if p != host)
    _save(backup_root, SyncConfig(config.device_id, peers))
    return True


def _run_remote(
    host: str,
    remote_args: str,
    stdin_bytes: bytes | None = None,
    *,
    timeout_s: float = _REMOTE_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    cmd = ["ssh", *_SSH_OPTS, "--", host, f"{_REMOTE_CSWAP} {remote_args}"]
    try:
        return subprocess.run(
            cmd,
            input=stdin_bytes,
            capture_output=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        raise SyncError("ssh not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise SyncError(f"{host}: remote cswap timed out") from None


def _remote_error(host: str, proc: subprocess.CompletedProcess) -> SyncError:
    detail = proc.stderr.decode(errors="replace").strip()
    tail = detail.splitlines()[-1] if detail else f"exit code {proc.returncode}"
    return SyncError(f"{host}: {tail}")


def pull_from_peer(switcher, host: str, *, force: bool = False) -> None:
    """Import the peer's accounts (remote ``cswap export -`` → local import)."""
    proc = _run_remote(host, "export -")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise _remote_error(host, proc)
    # import_accounts reads a path; keep the envelope 0600 inside the
    # backup root and always remove it.
    fd, tmp = tempfile.mkstemp(
        dir=switcher.backup_dir, prefix=".sync-pull-", suffix=".json"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(proc.stdout)
        import_accounts(switcher, tmp, force=force, heal_live=True)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def push_to_peer(
    switcher, host: str, *, force: bool = False, full: bool = False
) -> None:
    """Export local accounts and import them on the peer over stdin."""
    fd, tmp = tempfile.mkstemp(
        dir=switcher.backup_dir, prefix=".sync-push-", suffix=".json"
    )
    os.close(fd)
    try:
        # export's "Exported N account(s) to <tmp>" line (stderr) names a
        # temp file nobody should see; the peer's import summary is the real
        # output. Everything else export says (skip warnings) passes through.
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            export_accounts(switcher, tmp, full=full)
        for line in (out_buf.getvalue() + err_buf.getvalue()).splitlines():
            if not line.startswith("Exported "):
                print(line, file=sys.stderr)
        envelope = Path(tmp).read_bytes()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    import_args = "import -" + (" --force" if force else "")
    proc = _run_remote(host, f"{import_args} --heal-live", envelope)
    if proc.returncode == 2:
        # argparse exit code for an unknown flag: the peer runs an older
        # cswap without --heal-live. One extra round-trip, old behavior.
        print(dimmed(f"  {host}: no live-heal (older cswap?)"))
        proc = _run_remote(host, import_args, envelope)
    if proc.returncode != 0:
        raise _remote_error(host, proc)
    out = proc.stdout.decode(errors="replace").strip()
    for line in out.splitlines():
        print(f"  {dimmed(line)}")


GOSSIP_SCHEMA_VERSION = 1


def _identities(switcher) -> dict[str, tuple[str, str]]:
    """Slot → (email, organizationUuid) for every managed account."""
    data = switcher._get_sequence_data() or {}
    return {
        num: (acc.get("email") or "", acc.get("organizationUuid") or "")
        for num, acc in (data.get("accounts") or {}).items()
        if isinstance(acc, dict)
    }


def emit_usage(switcher) -> str:
    """The usage-gossip payload: this device's freshest measurements."""
    return json.dumps(
        {
            "schemaVersion": GOSSIP_SCHEMA_VERSION,
            "usage": switcher._usage_store.export_rows(),
        }
    )


def absorb_usage(switcher, payload: object) -> int:
    """Merge a peer's gossip payload; returns adopted-measurement count."""
    if (
        not isinstance(payload, dict)
        or payload.get("schemaVersion") != GOSSIP_SCHEMA_VERSION
        or not isinstance(payload.get("usage"), dict)
    ):
        raise SyncError("unrecognized usage-gossip payload")
    return switcher._usage_store.merge_remote_rows(
        payload["usage"], _identities(switcher)
    )


def gossip_usage(switcher, host: str, *, pull: bool, push: bool) -> None:
    """Trade usage measurements with a peer so neither re-polls what the
    other just fetched. Best-effort by design: a peer running plain
    claude-swap has no gossip verbs, and account sync must not fail on it."""
    try:
        if pull:
            proc = _run_remote(host, "sync emit-usage")
            if proc.returncode != 0:
                print(dimmed(f"  {host}: no usage gossip (older cswap?)"))
                return
            merged = absorb_usage(switcher, json.loads(proc.stdout.decode()))
            if merged:
                print(dimmed(f"  usage: adopted {merged} fresher measurement(s)"))
        if push:
            proc = _run_remote(
                host, "sync absorb-usage -", emit_usage(switcher).encode()
            )
            if proc.returncode != 0:
                print(dimmed(f"  {host}: no usage gossip (older cswap?)"))
    except (SyncError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        warning(f"  usage gossip with {host} skipped: {exc}")


# --- Active-account intent: broadcast on switch, gossip at sync time ------

_BROADCAST_TIMEOUT_S = 15


def _age(ts: float) -> str:
    delta = max(0.0, time.time() - ts)
    if delta < 90:
        return f"{int(delta)}s ago"
    if delta < 5400:
        return f"{int(delta / 60)}m ago"
    return f"{delta / 3600:.1f}h ago"


def emit_active(switcher) -> str:
    """This device's active-intent record (or null) as a gossip payload."""
    return json.dumps(
        {
            "schemaVersion": active_intent.INTENT_SCHEMA_VERSION,
            "intent": active_intent.load_intent(switcher.backup_dir),
        }
    )


def apply_active(switcher, payload: object, *, source: str = "") -> dict:
    """Consider a peer's active intent; follow it when it wins.

    Returns ``{"status", "reason", "detail"}`` — ``applied`` (switched and
    recorded), ``noop`` (nothing to do), or ``skipped`` (policy or state
    stopped us; not recorded, so a later sync retries). Every policy outcome
    is a normal return: only a malformed payload raises.
    """
    if not isinstance(payload, dict) or payload.get(
        "schemaVersion"
    ) != active_intent.INTENT_SCHEMA_VERSION:
        raise SyncError("unrecognized active-intent payload")
    if payload.get("intent") is None:
        return {"status": "noop", "reason": "no-intent", "detail": "peer has no intent"}
    intent = active_intent.validate_intent(payload)
    if intent is None:
        raise SyncError("unrecognized active-intent payload")

    backup_root = switcher.backup_dir
    email = intent["email"]
    config = load_sync_config(backup_root, create=False)
    if config.device_id and intent["originDeviceId"] == config.device_id:
        # Our own intent coming back around the fleet — the echo guard.
        return {"status": "noop", "reason": "own-intent", "detail": "originated here"}
    current = active_intent.load_intent(backup_root)
    if not active_intent.is_newer(intent, current):
        return {"status": "noop", "reason": "not-newer", "detail": "already have it"}

    from claude_swap.settings import load_sync_section_settings

    if not load_sync_section_settings(backup_root).follow_remote_switches:
        # Still adopt the record so this device relays it onward at its own
        # next sync — following and relaying are different decisions.
        active_intent.adopt_intent(backup_root, intent, source=source)
        return {
            "status": "skipped",
            "reason": "follow-disabled",
            "detail": "not following (sync.followRemoteSwitches=false)",
        }

    data = switcher._get_sequence_data() or {}
    slot = switcher._find_account_slot(data, email, intent["organizationUuid"])
    if slot is None:
        # Not recorded: once account sync delivers this identity, the same
        # intent applies cleanly on retry.
        return {
            "status": "skipped",
            "reason": "unknown-account",
            "detail": f"{email} is not managed here (run: cswap sync)",
        }
    current_slot = switcher.current_account_number()
    if current_slot == slot:
        active_intent.adopt_intent(backup_root, intent, source=source)
        return {"status": "noop", "reason": "already-active", "detail": f"already on {email}"}
    if current_slot is None and switcher.has_live_login():
        # A live login cswap doesn't manage was made deliberately out-of-band;
        # a remote intent must not clobber it.
        return {
            "status": "skipped",
            "reason": "unmanaged-live-login",
            "detail": "live login is not cswap-managed",
        }

    result = switcher.switch_to(slot, json_output=True, origin="remote")
    if not result or not (result.get("switched") or result.get("reason") == "activated"):
        detail = (result or {}).get("message", "switch did not complete")
        return {"status": "skipped", "reason": "switch-failed", "detail": detail}
    active_intent.adopt_intent(backup_root, intent, source=source)
    warnings = result.get("warnings") or []
    return {
        "status": "applied",
        "reason": "switched",
        "detail": f"switched to {email}"
        + (f" ({'; '.join(warnings)})" if warnings else ""),
    }


def broadcast_active(switcher, intent: dict, hosts: tuple[str, ...]) -> list[dict]:
    """Push a just-minted intent to each peer in parallel, best-effort.

    Runs right after a switch commits, so it is time-capped well below the
    normal remote timeout: an unreachable peer dies at SSH's ConnectTimeout
    and a wedged one at ``_BROADCAST_TIMEOUT_S``. Old peers (no verb) and
    unreachable ones degrade alike — they catch up at the next sync.
    """
    payload = json.dumps(
        {"schemaVersion": active_intent.INTENT_SCHEMA_VERSION, "intent": intent}
    ).encode()

    def push(host: str) -> dict:
        try:
            proc = _run_remote(
                host, "sync apply-active -", payload, timeout_s=_BROADCAST_TIMEOUT_S
            )
        except SyncError as exc:
            return {"host": host, "ok": False, "detail": str(exc)}
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace").strip()
            tail = stderr.splitlines()[-1] if stderr else "unreachable"
            return {"host": host, "ok": False, "detail": tail}
        out = proc.stdout.decode(errors="replace").strip()
        return {"host": host, "ok": True, "detail": out.splitlines()[-1] if out else ""}

    with ThreadPoolExecutor(max_workers=min(len(hosts), 4)) as pool:
        return list(pool.map(push, hosts))


def gossip_active(switcher, host: str, *, pull: bool, push: bool) -> None:
    """Trade active intent with a peer so switches survive unreachable
    devices: the Mac (inbound-unreachable) adopts remote switches when it
    initiates the sync, and a follow-off relay still hands intents onward.
    Best-effort like usage gossip — account sync must not fail on it."""
    try:
        if pull:
            proc = _run_remote(host, "sync emit-active")
            if proc.returncode != 0:
                print(dimmed(f"  {host}: no switch gossip (older cswap?)"))
                return
            res = apply_active(
                switcher, json.loads(proc.stdout.decode()), source=host
            )
            if res["status"] == "applied":
                intent = active_intent.load_intent(switcher.backup_dir) or {}
                age = _age(float(intent.get("ts", time.time())))
                print(
                    f"  {accent('following switch')} to {intent.get('email', '?')} "
                    f"from {host} ({age})"
                )
            elif res["status"] == "skipped":
                print(dimmed(f"  switch intent from {host}: {res['detail']}"))
        if push:
            intent = active_intent.load_intent(switcher.backup_dir)
            if intent is None:
                return
            proc = _run_remote(
                host,
                "sync apply-active -",
                json.dumps(
                    {
                        "schemaVersion": active_intent.INTENT_SCHEMA_VERSION,
                        "intent": intent,
                    }
                ).encode(),
            )
            if proc.returncode != 0:
                print(dimmed(f"  {host}: no switch gossip (older cswap?)"))
                return
            out = proc.stdout.decode(errors="replace").strip()
            if out and "noop" not in out:
                print(f"  {dimmed(out.splitlines()[-1])}")
    except (SyncError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        warning(f"  switch gossip with {host} skipped: {exc}")


def sync_peers(
    switcher,
    hosts: list[str],
    *,
    pull: bool = True,
    push: bool = True,
    force: bool = False,
    full: bool = False,
) -> int:
    """Sync each host in turn; one failing peer never blocks the rest.

    Returns the number of peers that failed.
    """
    failures = 0
    for host in hosts:
        print(f"{accent('Syncing')} {host}")
        try:
            if pull:
                pull_from_peer(switcher, host, force=force)
            if push:
                push_to_peer(switcher, host, force=force, full=full)
            gossip_usage(switcher, host, pull=pull, push=push)
            # After account pull/push so a just-delivered account resolves.
            gossip_active(switcher, host, pull=pull, push=push)
        except SyncError as exc:
            warning(f"  {exc}")
            failures += 1
    return failures
