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
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

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


def load_sync_config(backup_root: Path) -> SyncConfig:
    """Read sync.json, minting (and persisting) a device id on first use."""
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
    host: str, remote_args: str, stdin_bytes: bytes | None = None
) -> subprocess.CompletedProcess:
    cmd = ["ssh", *_SSH_OPTS, "--", host, f"{_REMOTE_CSWAP} {remote_args}"]
    try:
        return subprocess.run(
            cmd,
            input=stdin_bytes,
            capture_output=True,
            timeout=_REMOTE_TIMEOUT_S,
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
        import_accounts(switcher, tmp, force=force)
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
        # export's "Exported N account(s) to <tmp>" line names a temp file
        # nobody should see; the peer's import summary is the real output.
        with contextlib.redirect_stdout(io.StringIO()):
            export_accounts(switcher, tmp, full=full)
        envelope = Path(tmp).read_bytes()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    proc = _run_remote(host, "import -" + (" --force" if force else ""), envelope)
    if proc.returncode != 0:
        raise _remote_error(host, proc)
    out = proc.stdout.decode(errors="replace").strip()
    for line in out.splitlines():
        print(f"  {dimmed(line)}")


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
        except SyncError as exc:
            warning(f"  {exc}")
            failures += 1
    return failures
