"""heal: reactive fleet repair of dead credential lineages."""

import json
import subprocess
import time
from unittest.mock import patch

from claude_swap import heal, oauth


def _creds(refresh: str, expires_at: int) -> str:
    return json.dumps({
        "claudeAiOauth": {
            "accessToken": f"at-{refresh}",
            "refreshToken": refresh,
            "expiresAt": expires_at,
        }
    })


def _envelope(email: str, creds_text: str, org: str = "") -> bytes:
    return json.dumps({
        "version": 1,
        "accounts": [{
            "number": 1,
            "email": email,
            "organizationUuid": org,
            "credentials": json.loads(creds_text),
            "config": {"oauthAccount": {"emailAddress": email}},
        }],
    }).encode()


def _proc(rc=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr=stderr)


class HealSwitcher:
    """Just enough switcher surface for heal_from_peers/heal_all_dead."""

    def __init__(self, tmp_path, *, creds=None, kind="oauth", current=None):
        self.backup_dir = tmp_path
        (tmp_path / "cache").mkdir(exist_ok=True)
        self.lock_file = tmp_path / ".lock"
        self.creds = {"1": creds or _creds("rt-dead", 5000)}
        self.configs = {}
        self.kind = kind
        self.current = current  # (email, org) or None
        self.switch_calls = []
        self.cleared = []
        self.live_creds = None  # the live store (default profile) bytes
        self.session_heal_calls = []
        self.session_heal_result = True
        self.accounts = {"1": {"email": "a@x.com", "organizationUuid": ""}}
        self._usage_store = self

    # switcher surface
    def _get_sequence_data(self):
        return {"accounts": self.accounts}

    def _find_account_slot(self, data, email, org):
        for num, acc in data.get("accounts", {}).items():
            if acc["email"] == email and acc["organizationUuid"] == (org or ""):
                return num
        return None

    def account_kind_for(self, slot):
        return self.kind

    def read_account_credentials(self, slot, email):
        return self.creds.get(slot, "")

    def _write_account_credentials(self, slot, email, text):
        self.creds[slot] = text

    def _write_account_config(self, slot, email, text):
        self.configs[slot] = text

    def _get_current_account(self):
        return self.current

    def _read_credentials(self):
        return self.live_creds

    def live_session_pids_for(self, slot, email):
        return []

    def heal_session_profile(self, slot, email, org_uuid, creds_text, dead_fps):
        self.session_heal_calls.append((slot, email, org_uuid, creds_text))
        return self.session_heal_result

    def switch_to(self, slot, json_output=False, force=False, origin="manual"):
        self.switch_calls.append((slot, force, origin))
        return {"switched": True}

    # usage-store surface
    def clear_dead_token(self, nums, identities):
        self.cleared.extend(nums)

    def entries(self, identities):
        return {}


class TestAdoptionCriterion:
    def test_earlier_expiry_but_different_fingerprint_is_adopted(self, tmp_path):
        """THE load-bearing test: the dead copy can carry a LATER expiresAt
        than the working successor; fingerprint-differs must win anyway."""
        sw = HealSwitcher(tmp_path, creds=_creds("rt-dead", 9999))
        donor = _creds("rt-alive", 1000)  # earlier expiry, different lineage
        with patch.object(heal, "_run_remote",
                          return_value=_proc(0, _envelope("a@x.com", donor))):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed"
        assert sw.creds["1"] == donor
        assert sw.cleared == ["1"]

    def test_same_fingerprint_donor_refused(self, tmp_path):
        dead = _creds("rt-dead", 5000)
        sw = HealSwitcher(tmp_path, creds=dead)
        with patch.object(heal, "_run_remote",
                          return_value=_proc(0, _envelope("a@x.com", dead))):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "no-donor"
        assert sw.creds["1"] == dead

    def test_highest_expiry_donor_wins(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        donors = {
            "mm": _envelope("a@x.com", _creds("rt-b", 2000)),
            "ubuntu": _envelope("a@x.com", _creds("rt-c", 3000)),
        }
        with patch.object(
            heal, "_run_remote",
            side_effect=lambda host, *a, **k: _proc(0, donors[host]),
        ):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm", "ubuntu"))
        assert out.status == "healed" and out.host == "ubuntu"
        assert json.loads(sw.creds["1"])["claudeAiOauth"]["refreshToken"] == "rt-c"

    def test_ledgered_fingerprint_never_readopted(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        died_before = _creds("rt-died-once", 8000)
        heal.note_dead_fingerprint(
            tmp_path, "a@x.com", "", oauth.credential_fingerprint(died_before)
        )
        with patch.object(heal, "_run_remote",
                          return_value=_proc(0, _envelope("a@x.com", died_before))):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "no-donor"

    def test_quarantine_fingerprint_counts_as_dead(self, tmp_path):
        quarantined = _creds("rt-quarantined", 7000)
        (tmp_path / "autoswitch_state.json").write_text(json.dumps({
            "schemaVersion": 1,
            "quarantine": {"1": {
                "email": "a@x.com", "reason": "invalid_grant",
                "refreshTokenFingerprint": oauth.credential_fingerprint(quarantined),
            }},
        }))
        sw = HealSwitcher(tmp_path)
        with patch.object(heal, "_run_remote",
                          return_value=_proc(0, _envelope("a@x.com", quarantined))):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "no-donor"


class TestGatesAndBackoff:
    def test_no_peers_writes_no_state(self, tmp_path):
        """The ubuntu path: no outbound SSH, instant no-op, zero state."""
        sw = HealSwitcher(tmp_path)
        out = heal.heal_from_peers(sw, "a@x.com", "")
        assert out.status == "no-peers"
        assert not heal.heal_state_path(tmp_path).exists()

    def test_heal_on_death_toggle(self, tmp_path):
        from claude_swap.settings import set_setting

        set_setting(tmp_path, "sync.healOnDeath", "false")
        sw = HealSwitcher(tmp_path)
        out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "disabled"

    def test_api_key_skip(self, tmp_path):
        sw = HealSwitcher(tmp_path, kind="api_key")
        out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "skipped-api-key"

    def test_cooldown_after_failure_and_backoff_doubles(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        with patch.object(heal, "_run_remote", return_value=_proc(3)):
            assert heal.heal_from_peers(
                sw, "a@x.com", "", hosts=("mm",)
            ).status == "no-donor"
            # Immediately again: cooldown.
            assert heal.heal_from_peers(
                sw, "a@x.com", "", hosts=("mm",)
            ).status == "cooldown"
        assert heal._backoff_s(1) == 300.0
        assert heal._backoff_s(2) == 600.0
        assert heal._backoff_s(99) == heal.HEAL_BACKOFF_CAP_S

    def test_cooldown_expires(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        with patch.object(heal, "_run_remote", return_value=_proc(3)):
            heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
            with patch.object(heal.time, "time",
                              return_value=time.time() + 301):
                out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "no-donor"  # attempted again, not cooldown

    def test_raced_when_credential_changes_mid_heal(self, tmp_path):
        sw = HealSwitcher(tmp_path, creds=_creds("rt-dead", 5000))
        donor = _envelope("a@x.com", _creds("rt-alive", 9000))

        def remote_and_relogin(*a, **k):
            # A re-login lands while the SSH round-trip is in flight.
            sw.creds["1"] = _creds("rt-relogin", 9500)
            return _proc(0, donor)

        with patch.object(heal, "_run_remote", side_effect=remote_and_relogin):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "raced"
        assert json.loads(sw.creds["1"])["claudeAiOauth"]["refreshToken"] == "rt-relogin"


class TestLiveActivation:
    def test_live_identity_is_force_activated(self, tmp_path):
        sw = HealSwitcher(tmp_path, current=("a@x.com", ""))
        donor = _envelope("a@x.com", _creds("rt-alive", 9000))
        with patch.object(heal, "_run_remote", return_value=_proc(0, donor)):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed-live"
        assert sw.switch_calls == [("1", True, "remote-heal")]

    def test_running_session_is_reseeded_and_live_still_activated(self, tmp_path):
        """A running instance is a reason to repair, not to skip: the
        session profile is reseeded in place AND the live login is still
        activated — both surfaces held the dead generation."""
        sw = HealSwitcher(tmp_path, current=("a@x.com", ""))
        sw.live_session_pids_for = lambda slot, email: [1234]
        donor = _creds("rt-alive", 9000)
        with patch.object(heal, "_run_remote",
                          return_value=_proc(0, _envelope("a@x.com", donor))):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed-live"
        assert sw.switch_calls == [("1", True, "remote-heal")]
        assert sw.session_heal_calls == [("1", "a@x.com", "", donor)]
        assert "running session reseeded" in out.detail

    def test_running_session_with_alive_family_noted_not_reseeded(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        sw.live_session_pids_for = lambda slot, email: [1234]
        sw.session_heal_result = False  # switcher judged the family alive
        donor = _envelope("a@x.com", _creds("rt-alive", 9000))
        with patch.object(heal, "_run_remote", return_value=_proc(0, donor)):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed"
        assert "not dead; left as-is" in out.detail

    def test_live_login_with_working_credential_left_untouched(self, tmp_path):
        """An out-of-band re-login already fixed the running instance: its
        fresh grant is the one credential that provably works — activation
        must not replace it with the peer's copy."""
        sw = HealSwitcher(tmp_path, current=("a@x.com", ""))
        sw.live_creds = _creds("rt-relogin-fresh", 8000)
        donor = _envelope("a@x.com", _creds("rt-alive", 9000))
        with patch.object(heal, "_run_remote", return_value=_proc(0, donor)):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed"
        assert sw.switch_calls == []
        assert "working credential; left as-is" in out.detail

    def test_live_login_holding_the_dead_generation_is_activated(self, tmp_path):
        sw = HealSwitcher(tmp_path, current=("a@x.com", ""))
        sw.live_creds = sw.creds["1"]  # live store = the dead backup copy
        donor = _envelope("a@x.com", _creds("rt-alive", 9000))
        with patch.object(heal, "_run_remote", return_value=_proc(0, donor)):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed-live"
        assert sw.switch_calls == [("1", True, "remote-heal")]

    def test_live_login_already_on_healed_generation_not_reactivated(self, tmp_path):
        sw = HealSwitcher(tmp_path, current=("a@x.com", ""))
        donor = _creds("rt-alive", 9000)
        sw.live_creds = donor  # a concurrent repair already landed it
        with patch.object(heal, "_run_remote",
                          return_value=_proc(0, _envelope("a@x.com", donor))):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed"
        assert sw.switch_calls == []
        assert "already on the healed generation" in out.detail

    def test_activation_failure_still_reports_healed(self, tmp_path):
        sw = HealSwitcher(tmp_path, current=("a@x.com", ""))

        def broken_switch(*a, **k):
            raise RuntimeError("keychain sad")

        sw.switch_to = broken_switch
        donor = _envelope("a@x.com", _creds("rt-alive", 9000))
        with patch.object(heal, "_run_remote", return_value=_proc(0, donor)):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "healed"
        assert "activation failed" in out.detail


class TestPeerIteration:
    def test_rc3_peer_skipped_next_tried(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        donor = _envelope("a@x.com", _creds("rt-alive", 9000))
        procs = {"mm": _proc(3), "ubuntu": _proc(0, donor)}
        with patch.object(heal, "_run_remote",
                          side_effect=lambda h, *a, **k: procs[h]):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm", "ubuntu"))
        assert out.status == "healed" and out.host == "ubuntu"

    def test_unreachable_peer_skipped(self, tmp_path):
        sw = HealSwitcher(tmp_path)

        def flaky(host, *a, **k):
            if host == "mm":
                raise OSError("network down")
            return _proc(0, _envelope("a@x.com", _creds("rt-alive", 9000)))

        with patch.object(heal, "_run_remote", side_effect=flaky):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm", "ubuntu"))
        assert out.status == "healed"

    def test_api_key_entry_from_peer_never_adopted(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        env = json.dumps({
            "version": 1,
            "accounts": [{
                "number": 1, "email": "a@x.com", "organizationUuid": "",
                "credentials": "sk-ant-api03-xyz",
                "config": {},
            }],
        }).encode()
        with patch.object(heal, "_run_remote", return_value=_proc(0, env)):
            out = heal.heal_from_peers(sw, "a@x.com", "", hosts=("mm",))
        assert out.status == "no-donor"


class TestDeathHook:
    def test_on_death_ledgers_and_spawns_once(self, tmp_path):
        from claude_swap import sync as sync_mod

        sync_mod.add_peer(tmp_path, "mm")
        sw = HealSwitcher(tmp_path)
        fp = oauth.credential_fingerprint(sw.creds["1"])
        with patch.object(heal, "spawn_background_heal") as spawn:
            heal.on_death_detected(sw, [("1", "a@x.com", "", fp)])
        assert spawn.call_count == 1
        state = heal._read_state(tmp_path)
        fps = state["identities"]["a@x.com|"]["deadFingerprints"]
        assert fp in fps

    def test_on_death_ledgers_the_failed_fingerprint_not_the_slot(self, tmp_path):
        """THE race regression: a sync freshen replaces the slot's bytes while
        the doomed refresh is on the wire. The hook must ledger the
        fingerprint that failed — fingerprinting the slot's current (working,
        just-delivered) copy would mark the fleet's only live generation dead
        and block every future heal of it."""
        sw = HealSwitcher(tmp_path)
        dead_fp = oauth.credential_fingerprint(_creds("rt-consumed", 4000))
        sw.creds["1"] = _creds("rt-freshened-alive", 9000)  # freshen won
        with patch.object(heal, "spawn_background_heal"):
            heal.on_death_detected(sw, [("1", "a@x.com", "", dead_fp)])
        fps = heal._read_state(tmp_path)["identities"]["a@x.com|"][
            "deadFingerprints"
        ]
        assert dead_fp in fps
        assert oauth.credential_fingerprint(sw.creds["1"]) not in fps

    def test_on_death_respects_toggle(self, tmp_path):
        from claude_swap.settings import set_setting

        set_setting(tmp_path, "sync.healOnDeath", "false")
        sw = HealSwitcher(tmp_path)
        with patch.object(heal, "spawn_background_heal") as spawn:
            heal.on_death_detected(sw, [("1", "a@x.com", "", "sha256:dead")])
        assert not spawn.called

    def test_spawn_requires_peers(self, tmp_path):
        sw = HealSwitcher(tmp_path)
        with patch.object(heal.subprocess, "Popen") as popen:
            assert heal.spawn_background_heal(sw) is False
        assert not popen.called


class TestLiveFingerprint:
    def test_proven_live_fingerprint_leaves_the_ledger(self, tmp_path):
        (tmp_path / "cache").mkdir(exist_ok=True)
        heal.note_dead_fingerprint(tmp_path, "a@x.com", "", "sha256:poisoned")
        heal.note_live_fingerprint(tmp_path, "a@x.com", "", "sha256:poisoned")
        entry = heal._read_state(tmp_path)["identities"]["a@x.com|"]
        assert "sha256:poisoned" not in entry.get("deadFingerprints", {})
        assert entry["lastOutcome"] == "proven-live"

    def test_success_ends_the_heal_backoff(self, tmp_path):
        (tmp_path / "cache").mkdir(exist_ok=True)

        def fail(state):
            entry = state.setdefault("identities", {}).setdefault("a@x.com|", {})
            entry["consecutiveFailures"] = 5

        heal._mutate_state(tmp_path, fail)
        heal.note_live_fingerprint(tmp_path, "a@x.com", "", "sha256:whatever")
        entry = heal._read_state(tmp_path)["identities"]["a@x.com|"]
        assert entry["consecutiveFailures"] == 0

    def test_quiet_path_writes_nothing(self, tmp_path):
        """Nearly every poll succeeds against an empty ledger — that path
        must not create state or take the lock-write round-trip."""
        (tmp_path / "cache").mkdir(exist_ok=True)
        heal.note_live_fingerprint(tmp_path, "a@x.com", "", "sha256:fine")
        assert not heal.heal_state_path(tmp_path).exists()

    def test_other_dead_fingerprints_stay_ledgered(self, tmp_path):
        (tmp_path / "cache").mkdir(exist_ok=True)
        heal.note_dead_fingerprint(tmp_path, "a@x.com", "", "sha256:really-dead")
        heal.note_dead_fingerprint(tmp_path, "a@x.com", "", "sha256:poisoned")
        heal.note_live_fingerprint(tmp_path, "a@x.com", "", "sha256:poisoned")
        fps = heal._read_state(tmp_path)["identities"]["a@x.com|"][
            "deadFingerprints"
        ]
        assert "sha256:really-dead" in fps and "sha256:poisoned" not in fps
