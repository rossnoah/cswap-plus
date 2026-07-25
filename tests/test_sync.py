"""cswap sync: peer config, SSH transport plumbing, and the CLI verb."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from claude_swap import sync
from claude_swap.exceptions import SyncError


class FakeSwitcher:
    def __init__(self, backup_dir):
        self.backup_dir = backup_dir


def _proc(rc=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr=stderr)


class TestSyncConfig:
    def test_first_load_mints_and_persists_device_id(self, tmp_path):
        config = sync.load_sync_config(tmp_path)
        assert config.device_id
        assert config.peers == ()
        saved = json.loads((tmp_path / "sync.json").read_text())
        assert saved["deviceId"] == config.device_id
        # Stable across loads.
        assert sync.load_sync_config(tmp_path).device_id == config.device_id

    def test_add_and_remove_peer(self, tmp_path):
        assert sync.add_peer(tmp_path, "mm") is True
        assert sync.add_peer(tmp_path, "mm") is False
        assert sync.add_peer(tmp_path, "noah@ubuntu") is True
        assert sync.load_sync_config(tmp_path).peers == ("mm", "noah@ubuntu")
        assert sync.remove_peer(tmp_path, "mm") is True
        assert sync.remove_peer(tmp_path, "mm") is False
        assert sync.load_sync_config(tmp_path).peers == ("noah@ubuntu",)

    def test_option_shaped_host_rejected(self, tmp_path):
        with pytest.raises(SyncError):
            sync.add_peer(tmp_path, "-oProxyCommand=evil")
        with pytest.raises(SyncError):
            sync.validate_host("host name")

    def test_corrupt_file_tolerated(self, tmp_path):
        (tmp_path / "sync.json").write_text("{not json")
        config = sync.load_sync_config(tmp_path)
        assert config.device_id
        assert config.peers == ()


class TestPull:
    def test_pull_feeds_envelope_to_import_and_cleans_up(self, tmp_path):
        switcher = FakeSwitcher(tmp_path)
        envelope = b'{"version": 1, "accounts": []}'
        seen = {}

        def fake_import(sw, source, force=False, heal_live=False):
            seen["bytes"] = open(source, "rb").read()
            seen["force"] = force
            seen["heal_live"] = heal_live

        with patch.object(sync, "_run_remote", return_value=_proc(0, envelope)) as rr, \
             patch.object(sync, "import_accounts", side_effect=fake_import):
            sync.pull_from_peer(switcher, "mm", force=True)
        assert rr.call_args.args == ("mm", "export -")
        assert seen == {"bytes": envelope, "force": True, "heal_live": True}
        assert not list(tmp_path.glob(".sync-pull-*"))

    def test_pull_remote_failure_raises_with_stderr_tail(self, tmp_path):
        switcher = FakeSwitcher(tmp_path)
        proc = _proc(1, b"", b"warning\ncswap: command not found\n")
        with patch.object(sync, "_run_remote", return_value=proc):
            with pytest.raises(SyncError, match="mm: cswap: command not found"):
                sync.pull_from_peer(switcher, "mm")

    def test_pull_temp_removed_even_when_import_raises(self, tmp_path):
        switcher = FakeSwitcher(tmp_path)
        with patch.object(sync, "_run_remote", return_value=_proc(0, b"{}")), \
             patch.object(sync, "import_accounts", side_effect=SyncError("boom")):
            with pytest.raises(SyncError):
                sync.pull_from_peer(switcher, "mm")
        assert not list(tmp_path.glob(".sync-pull-*"))


class TestPush:
    def test_push_exports_and_pipes_to_remote_import(self, tmp_path, capsys):
        switcher = FakeSwitcher(tmp_path)

        def fake_export(sw, destination, account=None, full=False):
            with open(destination, "wb") as fh:
                fh.write(b"ENVELOPE")

        with patch.object(sync, "export_accounts", side_effect=fake_export), \
             patch.object(
                 sync, "_run_remote", return_value=_proc(0, b"Done: 1 imported\n")
             ) as rr:
            sync.push_to_peer(switcher, "ubuntu", force=True)
        host, remote_args = rr.call_args.args[:2]
        assert host == "ubuntu"
        assert remote_args == "import - --force --heal-live"
        assert rr.call_args.args[2] == b"ENVELOPE"
        assert "Done: 1 imported" in capsys.readouterr().out
        assert not list(tmp_path.glob(".sync-push-*"))

    def test_push_retries_without_heal_live_for_old_peer(self, tmp_path, capsys):
        switcher = FakeSwitcher(tmp_path)

        def fake_export(sw, destination, account=None, full=False):
            with open(destination, "wb") as fh:
                fh.write(b"ENVELOPE")

        # argparse on an old peer rejects the unknown flag with exit code 2;
        # the push must retry once with the plain form.
        procs = [_proc(2, b"", b"unrecognized arguments: --heal-live"),
                 _proc(0, b"Done: 1 imported\n")]
        with patch.object(sync, "export_accounts", side_effect=fake_export), \
             patch.object(sync, "_run_remote", side_effect=procs) as rr:
            sync.push_to_peer(switcher, "ubuntu")
        assert [c.args[1] for c in rr.call_args_list] == [
            "import - --heal-live", "import -",
        ]
        assert "no live-heal (older cswap?)" in capsys.readouterr().out

    def test_push_remote_failure_raises(self, tmp_path):
        switcher = FakeSwitcher(tmp_path)
        with patch.object(sync, "export_accounts"), \
             patch.object(sync, "_run_remote", return_value=_proc(255, b"", b"denied")):
            with pytest.raises(SyncError, match="ubuntu: denied"):
                sync.push_to_peer(switcher, "ubuntu")


class TestSyncPeers:
    def test_failure_does_not_block_next_peer(self, tmp_path, capsys):
        switcher = FakeSwitcher(tmp_path)
        calls = []

        def fake_pull(sw, host, force=False):
            if host == "bad":
                raise SyncError("bad: unreachable")
            calls.append(("pull", host))

        def fake_push(sw, host, force=False, full=False):
            calls.append(("push", host))

        with patch.object(sync, "pull_from_peer", side_effect=fake_pull), \
             patch.object(sync, "push_to_peer", side_effect=fake_push), \
             patch.object(sync, "gossip_usage"), \
             patch.object(sync, "gossip_active"):
            failures = sync.sync_peers(switcher, ["bad", "good"])
        assert failures == 1
        assert calls == [("pull", "good"), ("push", "good")]
        assert "bad: unreachable" in capsys.readouterr().out

    def test_direction_flags(self, tmp_path):
        switcher = FakeSwitcher(tmp_path)
        with patch.object(sync, "pull_from_peer") as pull, \
             patch.object(sync, "push_to_peer") as push, \
             patch.object(sync, "gossip_usage"), \
             patch.object(sync, "gossip_active"):
            sync.sync_peers(switcher, ["mm"], push=False)
            assert pull.called and not push.called
            pull.reset_mock()
            sync.sync_peers(switcher, ["mm"], pull=False)
            assert push.called and not pull.called


class TestRunRemote:
    def test_command_shape_pins_path_and_batch_mode(self):
        with patch.object(sync.subprocess, "run", return_value=_proc()) as run:
            sync._run_remote("mm", "export -")
        cmd = run.call_args.args[0]
        assert cmd[0] == "ssh"
        assert "BatchMode=yes" in cmd
        assert cmd[-2] == "mm"
        assert cmd[-1] == 'PATH="$HOME/.local/bin:$PATH" cswap export -'
        assert "--" in cmd

    def test_timeout_becomes_sync_error(self):
        err = subprocess.TimeoutExpired(cmd="ssh", timeout=1)
        with patch.object(sync.subprocess, "run", side_effect=err):
            with pytest.raises(SyncError, match="timed out"):
                sync._run_remote("mm", "export -")


class TestSyncCli:
    def _run(self, argv, tmp_path, capsys):
        from claude_swap import cli

        fake = SimpleNamespace(
            backup_dir=tmp_path, _is_running_in_container=lambda: False
        )
        code = 0
        with patch.object(cli, "ClaudeAccountSwitcher", return_value=fake):
            try:
                cli._sync_command(argv)
            except SystemExit as exc:
                code = exc.code or 0
        out, err = capsys.readouterr()
        return code, out, err

    def test_add_list_remove_flow(self, tmp_path, capsys):
        code, out, _ = self._run(["add", "mm"], tmp_path, capsys)
        assert code == 0 and "mm" in out
        code, out, _ = self._run(["list"], tmp_path, capsys)
        assert code == 0 and "mm" in out
        code, out, _ = self._run(["remove", "mm"], tmp_path, capsys)
        assert code == 0
        code, out, _ = self._run(["list"], tmp_path, capsys)
        assert "No sync peers" in out

    def test_bare_sync_without_peers_errors(self, tmp_path, capsys):
        code, _, err = self._run([], tmp_path, capsys)
        assert code == 1
        assert "no sync peers" in err.lower()

    def test_bare_sync_uses_saved_peers(self, tmp_path, capsys):
        sync.add_peer(tmp_path, "mm")
        from claude_swap import cli

        with patch.object(sync, "sync_peers", return_value=0) as sp:
            code, _, _ = self._run([], tmp_path, capsys)
        assert code == 0
        assert sp.call_args.args[1] == ["mm"]
        kwargs = sp.call_args.kwargs
        assert kwargs["pull"] is True and kwargs["push"] is True

    def test_pull_flag_disables_push(self, tmp_path, capsys):
        sync.add_peer(tmp_path, "mm")
        with patch.object(sync, "sync_peers", return_value=0) as sp:
            self._run(["--pull"], tmp_path, capsys)
        assert sp.call_args.kwargs["push"] is False
        assert sp.call_args.kwargs["pull"] is True

    def test_failures_exit_nonzero(self, tmp_path, capsys):
        sync.add_peer(tmp_path, "mm")
        with patch.object(sync, "sync_peers", return_value=1):
            code, _, _ = self._run([], tmp_path, capsys)
        assert code == 1

    def test_verb_rejects_sync_options(self, tmp_path, capsys):
        code, _, err = self._run(["add", "mm", "--force"], tmp_path, capsys)
        assert code == 1
        assert "takes no sync options" in err


class IntentSwitcher:
    """Fake switcher with enough surface for apply_active/gossip_active."""

    def __init__(self, backup_dir, accounts=None, current=None, live=None):
        self.backup_dir = backup_dir
        # accounts: {slot: (email, org)}; current: active slot; live: bool
        self.accounts = accounts or {"1": ("a@x.com", ""), "2": ("b@x.com", "")}
        self.current = current
        self.live = live if live is not None else current is not None
        self.switch_calls = []
        self.switch_result = {"switched": True, "warnings": []}

    def _get_sequence_data(self):
        return {
            "accounts": {
                num: {"email": e, "organizationUuid": org}
                for num, (e, org) in self.accounts.items()
            }
        }

    def _find_account_slot(self, data, email, org_uuid):
        for num, acc in data.get("accounts", {}).items():
            if acc["email"] == email and acc["organizationUuid"] == (org_uuid or ""):
                return num
        return None

    def current_account_number(self):
        return self.current

    def has_live_login(self):
        return self.live

    def switch_to(self, slot, json_output=False, force=False, origin="manual"):
        self.switch_calls.append((slot, force, origin))
        return dict(self.switch_result)


def _intent(email="b@x.com", ts=100.0, device="peer-dev", kind="manual"):
    return {
        "email": email, "organizationUuid": "", "ts": ts,
        "originDeviceId": device, "originKind": kind, "adoptedFrom": None,
    }


def _payload(intent):
    return {"schemaVersion": 1, "intent": intent}


class TestApplyActive:
    def test_malformed_payload_raises(self, tmp_path):
        with pytest.raises(SyncError):
            sync.apply_active(IntentSwitcher(tmp_path), {"schemaVersion": 7})
        with pytest.raises(SyncError):
            sync.apply_active(
                IntentSwitcher(tmp_path), _payload({"email": ""})
            )

    def test_null_intent_is_noop(self, tmp_path):
        res = sync.apply_active(IntentSwitcher(tmp_path), _payload(None))
        assert res["status"] == "noop" and res["reason"] == "no-intent"

    def test_own_intent_echo_guard(self, tmp_path):
        config = sync.load_sync_config(tmp_path)  # mints deviceId
        sw = IntentSwitcher(tmp_path, current="1")
        res = sync.apply_active(
            sw, _payload(_intent(device=config.device_id))
        )
        assert res["reason"] == "own-intent"
        assert sw.switch_calls == []

    def test_not_newer_lww_noop(self, tmp_path):
        from claude_swap import active_intent

        active_intent.adopt_intent(tmp_path, _intent(ts=200.0), source="mm")
        sw = IntentSwitcher(tmp_path, current="1")
        res = sync.apply_active(sw, _payload(_intent(ts=150.0)))
        assert res["reason"] == "not-newer"
        assert sw.switch_calls == []

    def test_follow_disabled_adopts_but_never_switches(self, tmp_path):
        from claude_swap import active_intent
        from claude_swap.settings import set_setting

        set_setting(tmp_path, "sync.followRemoteSwitches", "false")
        sw = IntentSwitcher(tmp_path, current="1")
        res = sync.apply_active(sw, _payload(_intent()), source="mm")
        assert res["status"] == "skipped"
        assert res["reason"] == "follow-disabled"
        assert sw.switch_calls == []
        # Adopted anyway: this device relays the intent onward at sync time.
        stored = active_intent.load_intent(tmp_path)
        assert stored["ts"] == 100.0 and stored["adoptedFrom"] == "mm"

    def test_unknown_account_skipped_and_not_recorded(self, tmp_path):
        from claude_swap import active_intent

        sw = IntentSwitcher(tmp_path, current="1")
        res = sync.apply_active(sw, _payload(_intent(email="who@x.com")))
        assert res["reason"] == "unknown-account"
        assert active_intent.load_intent(tmp_path) is None  # retries later

    def test_already_active_adopts_as_noop(self, tmp_path):
        from claude_swap import active_intent

        sw = IntentSwitcher(tmp_path, current="2")
        res = sync.apply_active(sw, _payload(_intent()), source="mm")
        assert res["status"] == "noop" and res["reason"] == "already-active"
        assert sw.switch_calls == []
        assert active_intent.load_intent(tmp_path)["ts"] == 100.0

    def test_unmanaged_live_login_blocks(self, tmp_path):
        from claude_swap import active_intent

        sw = IntentSwitcher(tmp_path, current=None, live=True)
        res = sync.apply_active(sw, _payload(_intent()))
        assert res["reason"] == "unmanaged-live-login"
        assert sw.switch_calls == []
        assert active_intent.load_intent(tmp_path) is None

    def test_applied_switches_with_remote_origin_and_records(self, tmp_path):
        from claude_swap import active_intent

        sw = IntentSwitcher(tmp_path, current="1")
        res = sync.apply_active(sw, _payload(_intent()), source="mm")
        assert res["status"] == "applied"
        assert sw.switch_calls == [("2", False, "remote")]
        assert active_intent.load_intent(tmp_path)["originDeviceId"] == "peer-dev"

    def test_switch_failure_not_recorded(self, tmp_path):
        from claude_swap import active_intent

        sw = IntentSwitcher(tmp_path, current="1")
        sw.switch_result = {"switched": False, "reason": "no-valid-target",
                            "message": "nope"}
        res = sync.apply_active(sw, _payload(_intent()))
        assert res["reason"] == "switch-failed"
        assert active_intent.load_intent(tmp_path) is None


class TestBroadcastActive:
    def test_parallel_results_include_failures(self, tmp_path):
        sw = IntentSwitcher(tmp_path)

        def fake_remote(host, args, stdin=None, timeout_s=None):
            assert args == "sync apply-active -"
            assert json.loads(stdin)["intent"]["email"] == "b@x.com"
            if host == "down":
                return _proc(255, b"", b"ssh: connect refused\n")
            return _proc(0, b"switch intent: applied \xe2\x80\x94 ok\n")

        with patch.object(sync, "_run_remote", side_effect=fake_remote):
            results = sync.broadcast_active(sw, _intent(), ("mm", "down"))
        by_host = {r["host"]: r for r in results}
        assert by_host["mm"]["ok"] is True
        assert by_host["down"]["ok"] is False
        assert "refused" in by_host["down"]["detail"]

    def test_ssh_missing_degrades(self, tmp_path):
        sw = IntentSwitcher(tmp_path)
        with patch.object(sync, "_run_remote",
                          side_effect=SyncError("ssh not found on PATH")):
            results = sync.broadcast_active(sw, _intent(), ("mm",))
        assert results[0]["ok"] is False


class TestGossipActive:
    def test_pull_applies_and_reports(self, tmp_path, capsys):
        sw = IntentSwitcher(tmp_path, current="1")
        payload = json.dumps(_payload(_intent(ts=100.0))).encode()
        with patch.object(sync, "_run_remote", return_value=_proc(0, payload)):
            sync.gossip_active(sw, "mm", pull=True, push=False)
        assert sw.switch_calls == [("2", False, "remote")]
        assert "following switch" in capsys.readouterr().out

    def test_old_peer_degrades_quietly(self, tmp_path, capsys):
        sw = IntentSwitcher(tmp_path, current="1")
        with patch.object(sync, "_run_remote", return_value=_proc(1, b"", b"")):
            sync.gossip_active(sw, "mm", pull=True, push=True)
        out = capsys.readouterr().out
        assert "older cswap" in out

    def test_push_skipped_without_local_intent(self, tmp_path):
        sw = IntentSwitcher(tmp_path)
        with patch.object(sync, "_run_remote") as rr:
            sync.gossip_active(sw, "mm", pull=False, push=True)
        assert not rr.called

    def test_sync_peers_runs_account_sync_before_intent(self, tmp_path):
        sw = IntentSwitcher(tmp_path)
        order = []
        with patch.object(sync, "pull_from_peer",
                          side_effect=lambda *a, **k: order.append("pull")), \
             patch.object(sync, "push_to_peer",
                          side_effect=lambda *a, **k: order.append("push")), \
             patch.object(sync, "gossip_usage",
                          side_effect=lambda *a, **k: order.append("usage")), \
             patch.object(sync, "gossip_active",
                          side_effect=lambda *a, **k: order.append("active")):
            sync.sync_peers(sw, ["mm"])
        assert order.index("active") > order.index("pull")
        assert order.index("active") > order.index("push")


class TestIntentLoopScenario:
    def test_three_device_fleet_converges_without_echo(self, tmp_path):
        """Mac switches; mm and ubuntu follow; later syncs are all no-ops."""
        from claude_swap import active_intent

        roots = {}
        devices = {}
        for name in ("mac", "mm", "ubuntu"):
            root = tmp_path / name
            root.mkdir()
            roots[name] = root
            devices[name] = sync.load_sync_config(root).device_id

        # Mac mints the intent (as _announce_switch would).
        intent = active_intent.record_local_intent(
            roots["mac"], email="b@x.com", org_uuid="",
            device_id=devices["mac"], kind="manual",
        )

        applies = []

        def apply_on(name, source):
            sw = IntentSwitcher(
                roots[name],
                current="2" if active_intent.load_intent(roots[name]) else "1",
            )
            res = sync.apply_active(sw, _payload(intent), source=source)
            applies.append((name, res["status"], len(sw.switch_calls)))
            return res

        # Push from the Mac to both peers: both switch exactly once.
        assert apply_on("mm", "mac")["status"] == "applied"
        assert apply_on("ubuntu", "mac")["status"] == "applied"
        # mm later syncs with ubuntu: LWW makes both directions no-ops.
        assert apply_on("ubuntu", "mm")["status"] == "noop"
        assert apply_on("mm", "ubuntu")["status"] == "noop"
        # The Mac pulls its own intent back: echo guard.
        mac_sw = IntentSwitcher(roots["mac"], current="2")
        res = sync.apply_active(mac_sw, _payload(intent), source="mm")
        assert res["reason"] == "own-intent"
        # Exactly two switches happened fleet-wide.
        assert sum(n for _, _, n in applies) == 2


class AutosyncSwitcher(FakeSwitcher):
    def __init__(self, backup_dir):
        super().__init__(backup_dir)
        (backup_dir / "cache").mkdir(exist_ok=True)


class TestMaybeAutosync:
    def _armed(self, tmp_path):
        sync.add_peer(tmp_path, "mm")
        return AutosyncSwitcher(tmp_path)

    def test_toggle_off_is_instant_false(self, tmp_path):
        from claude_swap.settings import set_setting

        set_setting(tmp_path, "sync.autoSync", "false")
        sw = self._armed(tmp_path)
        with patch.object(sync, "sync_peers") as sp:
            assert sync.maybe_autosync(sw, source="test") is False
        assert not sp.called
        assert sync.autosync_due(sw) is False

    def test_no_peers_is_false(self, tmp_path):
        sw = AutosyncSwitcher(tmp_path)
        with patch.object(sync, "sync_peers") as sp:
            assert sync.maybe_autosync(sw, source="test") is False
        assert not sp.called

    def test_runs_then_throttles(self, tmp_path):
        sw = self._armed(tmp_path)
        with patch.object(sync, "sync_peers", return_value=0) as sp:
            assert sync.maybe_autosync(sw, source="test") is True
            assert sp.call_args.kwargs["quiet"] is True
            # Second call inside the interval: stamp throttles it.
            assert sync.maybe_autosync(sw, source="test") is False
        assert sp.call_count == 1
        stamp = json.loads((tmp_path / "cache" / "autosync.json").read_text())
        assert stamp["source"] == "test"

    def test_runs_heal_pass(self, tmp_path):
        sw = self._armed(tmp_path)
        with patch.object(sync, "sync_peers", return_value=0), \
             patch("claude_swap.heal.heal_all_dead") as had:
            sync.maybe_autosync(sw, source="test")
        assert had.called

    def test_spawn_only_when_due(self, tmp_path):
        sw = self._armed(tmp_path)
        with patch.object(sync.subprocess, "Popen") as popen:
            assert sync.spawn_background_autosync(sw, source="tui") is True
            args = popen.call_args.args[0]
            assert args[-3:] == ["sync", "--auto"] or args[-2:] == ["sync", "--auto"]
        # Stamp a fresh run: not due, no spawn.
        from claude_swap.settings import atomic_write_json
        import time as _time

        atomic_write_json(
            tmp_path / "cache" / "autosync.json",
            {"timestamp": _time.time(), "pid": 1},
        )
        with patch.object(sync.subprocess, "Popen") as popen:
            assert sync.spawn_background_autosync(sw, source="tui") is False
        assert not popen.called


class TestEmitCredential:
    def _switcher(self, tmp_path, kind="oauth"):
        sw = IntentSwitcher(tmp_path)
        sw.account_kind_for = lambda slot: kind
        return sw

    def test_unknown_identity_returns_none(self, tmp_path):
        sw = self._switcher(tmp_path)
        assert sync.emit_credential(
            sw, {"email": "who@x.com", "organizationUuid": ""}
        ) is None

    def test_api_key_returns_none(self, tmp_path):
        sw = self._switcher(tmp_path, kind="api_key")
        assert sync.emit_credential(
            sw, {"email": "a@x.com", "organizationUuid": ""}
        ) is None

    def test_known_identity_exports_slim_envelope(self, tmp_path):
        sw = self._switcher(tmp_path)
        with patch.object(sync, "export_accounts",
                          side_effect=lambda s, dest, account=None, full=False:
                          print(json.dumps({"version": 1, "account": account}))):
            out = sync.emit_credential(
                sw, {"email": "a@x.com", "organizationUuid": ""}
            )
        assert json.loads(out)["account"] == "1"

    def test_malformed_request_raises(self, tmp_path):
        sw = self._switcher(tmp_path)
        with pytest.raises(SyncError):
            sync.emit_credential(sw, {"email": ""})
        with pytest.raises(SyncError):
            sync.emit_credential(sw, ["nope"])


class TestSyncAutoCli:
    _run = TestSyncCli._run

    def test_auto_on_off_status(self, tmp_path, capsys):
        from claude_swap.settings import load_sync_section_settings

        code, out, _ = self._run(["auto", "off"], tmp_path, capsys)
        assert code == 0
        assert load_sync_section_settings(tmp_path).auto_sync is False
        code, out, _ = self._run(["auto", "on"], tmp_path, capsys)
        assert code == 0
        assert load_sync_section_settings(tmp_path).auto_sync is True
        code, out, _ = self._run(["auto"], tmp_path, capsys)
        assert code == 0
        assert "auto-sync: on" in out
        assert "schedule:" in out

    def test_auto_bad_mode_errors(self, tmp_path, capsys):
        code, _, err = self._run(["auto", "sideways"], tmp_path, capsys)
        assert code == 1
        assert "on|off|status" in err

    def test_auto_flag_routes_through_maybe_autosync(self, tmp_path, capsys):
        with patch.object(sync, "maybe_autosync", return_value=False) as ma:
            code, out, _ = self._run(["--auto"], tmp_path, capsys)
        assert code == 0
        assert ma.call_args.kwargs["source"] == "schedule"
        assert out == ""  # silent when nothing to do

    def test_install_schedule_no_peers_polite_exit_0(self, tmp_path, capsys):
        code, out, _ = self._run(["install-schedule"], tmp_path, capsys)
        assert code == 0
        assert "inbound pushes" in out


class TestInstallScheduleMacos:
    def test_writes_plist_and_loads(self, tmp_path, capsys, monkeypatch):
        if sync.sys.platform != "darwin":
            pytest.skip("launchd path is macOS-only")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        sync.add_peer(tmp_path, "mm")
        sw = SimpleNamespace(backup_dir=tmp_path)
        with patch.object(sync.subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0)) as run:
            sync.install_schedule(sw)
        plist = tmp_path / "Library" / "LaunchAgents" / "com.claude-swap.sync.plist"
        assert plist.exists()
        text = plist.read_text()
        assert "<string>sync</string>" in text
        assert "<string>--auto</string>" in text
        assert "StartInterval" in text
        assert any("launchctl" in c.args[0][0] for c in run.call_args_list)
        assert sync.schedule_installed() is True
        with patch.object(sync.subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0)):
            sync.uninstall_schedule()
        assert not plist.exists()
