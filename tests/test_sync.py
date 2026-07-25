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

        def fake_import(sw, source, force=False):
            seen["bytes"] = open(source, "rb").read()
            seen["force"] = force

        with patch.object(sync, "_run_remote", return_value=_proc(0, envelope)) as rr, \
             patch.object(sync, "import_accounts", side_effect=fake_import):
            sync.pull_from_peer(switcher, "mm", force=True)
        assert rr.call_args.args == ("mm", "export -")
        assert seen == {"bytes": envelope, "force": True}
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
        assert remote_args == "import - --force"
        assert rr.call_args.args[2] == b"ENVELOPE"
        assert "Done: 1 imported" in capsys.readouterr().out
        assert not list(tmp_path.glob(".sync-push-*"))

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
             patch.object(sync, "push_to_peer", side_effect=fake_push):
            failures = sync.sync_peers(switcher, ["bad", "good"])
        assert failures == 1
        assert calls == [("pull", "good"), ("push", "good")]
        assert "bad: unreachable" in capsys.readouterr().out

    def test_direction_flags(self, tmp_path):
        switcher = FakeSwitcher(tmp_path)
        with patch.object(sync, "pull_from_peer") as pull, \
             patch.object(sync, "push_to_peer") as push:
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
