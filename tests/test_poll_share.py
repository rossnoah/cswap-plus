"""Cross-device poll-budget sharing and usage gossip."""

import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from claude_swap import sync
from claude_swap.exceptions import SyncError
from claude_swap.poll_policy import (
    CANDIDATE_DEFAULT_INTERVAL_S,
    RESET_SLACK_S,
    plan_after_fetch,
    poll_phase,
)
from claude_swap.settings import load_poll_settings
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import UsageStore


def _iso_in(seconds: float) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).isoformat()


class TestPollPhase:
    def test_stable_and_in_range(self):
        a = poll_phase("dev1", ("a@x.com", "org"))
        assert a == poll_phase("dev1", ("a@x.com", "org"))
        assert 0.0 <= a < 1.0

    def test_varies_by_device_and_account(self):
        identity = ("a@x.com", "org")
        assert poll_phase("dev1", identity) != poll_phase("dev2", identity)
        assert poll_phase("dev1", identity) != poll_phase("dev1", ("b@x.com", "org"))


class TestShareFactorPlanning:
    def _plan(self, **kwargs):
        now = time.time()
        defaults = dict(
            prev_interval_s=None,
            prev_usage=None,
            new_usage=None,
            is_active=False,
            threshold=90.0,
            models=(),
            recent_429=False,
            now=now,
            rng=lambda: 0.5,
        )
        defaults.update(kwargs)
        return now, plan_after_fetch(**defaults)

    def test_share_multiplies_interval(self):
        now, (next_poll, interval) = self._plan(share_factor=3.0, phase=0.5)
        assert interval == CANDIDATE_DEFAULT_INTERVAL_S * 3
        assert next_poll == pytest.approx(now + interval, abs=1.0)

    def test_share_one_is_identity(self):
        _, (_, base_interval) = self._plan()
        _, (_, shared) = self._plan(share_factor=1.0)
        assert shared == base_interval == CANDIDATE_DEFAULT_INTERVAL_S

    def test_phase_replaces_random_jitter(self):
        # rng would explode if consulted; the phase draw must win.
        def boom():
            raise AssertionError("rng consulted despite phase")

        _, (next_a, _) = self._plan(share_factor=2.0, phase=0.0, rng=boom)
        _, (next_b, _) = self._plan(share_factor=2.0, phase=1.0, rng=boom)
        assert next_b > next_a  # opposite jitter extremes spread apart

    def test_reset_clamp_still_bounds_shared_interval(self):
        usage = {"five_hour": {"pct": 50.0, "resets_at": _iso_in(400)}}
        now, (next_poll, interval) = self._plan(
            new_usage=usage, share_factor=8.0, phase=0.5
        )
        assert interval == CANDIDATE_DEFAULT_INTERVAL_S * 8
        assert next_poll <= now + 400 + RESET_SLACK_S + 2.0


class TestPollSettings:
    def test_default_is_auto(self, tmp_path):
        assert load_poll_settings(tmp_path).budget_share == 0

    def test_explicit_value(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            json.dumps({"poll": {"budgetShare": 4}})
        )
        assert load_poll_settings(tmp_path).budget_share == 4

    @pytest.mark.parametrize("bad", ["3", 3.5, True, -1, 17])
    def test_invalid_falls_back(self, tmp_path, bad):
        (tmp_path / "settings.json").write_text(
            json.dumps({"poll": {"budgetShare": bad}})
        )
        assert load_poll_settings(tmp_path).budget_share == 0


class TestPollShareInputs:
    def _inputs(self, tmp_path):
        fake = SimpleNamespace(backup_dir=tmp_path, _poll_share_cache=None)
        return ClaudeAccountSwitcher._poll_share_inputs(fake)

    def test_no_config_means_fleet_of_one(self, tmp_path):
        assert self._inputs(tmp_path) == (1, "")
        assert not (tmp_path / "sync.json").exists()  # read never creates it

    def test_auto_counts_sync_peers(self, tmp_path):
        sync.add_peer(tmp_path, "mm")
        sync.add_peer(tmp_path, "ubuntu")
        share, device_id = self._inputs(tmp_path)
        assert share == 3
        assert device_id == sync.load_sync_config(tmp_path).device_id

    def test_explicit_setting_overrides_peer_count(self, tmp_path):
        sync.add_peer(tmp_path, "mm")
        (tmp_path / "settings.json").write_text(
            json.dumps({"poll": {"budgetShare": 5}})
        )
        assert self._inputs(tmp_path)[0] == 5


def _store(tmp_path) -> UsageStore:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    return UsageStore(cache)


def _row(email, org, fetched_at, pct=42.0, **extra):
    row = {
        "email": email,
        "organizationUuid": org,
        "lastGood": {"five_hour": {"pct": pct}},
        "fetchedAt": fetched_at,
    }
    row.update(extra)
    return row


class TestGossipRows:
    def test_export_keys_by_identity_and_skips_unmeasured(self, tmp_path):
        store = _store(tmp_path)
        store._write_rows(
            {
                "1": _row("a@x.com", "org-a", 1000.0),
                "2": {"email": "b@x.com", "organizationUuid": "", "lastGood": None},
            }
        )
        out = store.export_rows()
        assert set(out) == {"a@x.com|org-a"}
        assert out["a@x.com|org-a"]["fetchedAt"] == 1000.0

    def test_export_duplicate_identity_keeps_freshest(self, tmp_path):
        store = _store(tmp_path)
        store._write_rows(
            {
                "1": _row("a@x.com", "", 1000.0, pct=10.0),
                "2": _row("a@x.com", "", 2000.0, pct=20.0),
            }
        )
        out = store.export_rows()
        assert out["a@x.com|"]["lastGood"]["five_hour"]["pct"] == 20.0

    def test_merge_adopts_only_fresher(self, tmp_path):
        store = _store(tmp_path)
        store._write_rows(
            {
                "1": _row("a@x.com", "", 1000.0, pct=10.0, nextPollAt=1234.0),
                "2": _row("b@x.com", "", 5000.0, pct=50.0),
            }
        )
        identities = {"1": ("a@x.com", ""), "2": ("b@x.com", "")}
        merged = store.merge_remote_rows(
            {
                "a@x.com|": {
                    "fetchedAt": 2000.0,
                    "lastGood": {"five_hour": {"pct": 33.0}},
                    "last429At": 1500.0,
                },
                "b@x.com|": {
                    "fetchedAt": 4000.0,  # staler than ours
                    "lastGood": {"five_hour": {"pct": 99.0}},
                },
            },
            identities,
        )
        assert merged == 1
        rows = store._read_rows()
        assert rows["1"]["lastGood"]["five_hour"]["pct"] == 33.0
        assert rows["1"]["fetchedAt"] == 2000.0
        assert rows["1"]["last429At"] == 1500.0
        assert rows["1"]["nextPollAt"] == 1234.0  # plans stay local
        assert rows["2"]["lastGood"]["five_hour"]["pct"] == 50.0

    def test_merge_seeds_never_fetched_slot(self, tmp_path):
        store = _store(tmp_path)
        merged = store.merge_remote_rows(
            {
                "a@x.com|": {
                    "fetchedAt": 2000.0,
                    "lastGood": {"five_hour": {"pct": 12.0}},
                }
            },
            {"1": ("a@x.com", "")},
        )
        assert merged == 1
        assert store._read_rows()["1"]["lastGood"]["five_hour"]["pct"] == 12.0

    def test_merge_ignores_unknown_identities_and_junk(self, tmp_path):
        store = _store(tmp_path)
        store._write_rows({"1": _row("a@x.com", "", 1000.0)})
        merged = store.merge_remote_rows(
            {
                "stranger@y.com|": {"fetchedAt": 9000.0, "lastGood": {}},
                "a@x.com|": {"fetchedAt": "soon", "lastGood": {}},
            },
            {"1": ("a@x.com", "")},
        )
        assert merged == 0


class TestGossipTransport:
    def _switcher(self, tmp_path, accounts):
        return SimpleNamespace(
            backup_dir=tmp_path,
            _usage_store=_store(tmp_path),
            _get_sequence_data=lambda: {"accounts": accounts},
        )

    def test_emit_absorb_roundtrip(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        sender = self._switcher(
            tmp_path / "a", {"1": {"email": "a@x.com", "organizationUuid": ""}}
        )
        sender._usage_store._write_rows({"1": _row("a@x.com", "", 3000.0, pct=77.0)})
        receiver = self._switcher(
            tmp_path / "b", {"4": {"email": "a@x.com", "organizationUuid": ""}}
        )

        merged = sync.absorb_usage(receiver, json.loads(sync.emit_usage(sender)))
        assert merged == 1
        row = receiver._usage_store._read_rows()["4"]  # local slot, not sender's
        assert row["lastGood"]["five_hour"]["pct"] == 77.0

    def test_absorb_rejects_junk_payload(self, tmp_path):
        receiver = self._switcher(tmp_path, {})
        with pytest.raises(SyncError):
            sync.absorb_usage(receiver, {"schemaVersion": 99})

    def test_gossip_tolerates_peer_without_verbs(self, tmp_path, capsys):
        switcher = self._switcher(tmp_path, {})
        import subprocess

        proc = subprocess.CompletedProcess([], 2, stdout=b"", stderr=b"usage: ...")
        with patch.object(sync, "_run_remote", return_value=proc):
            sync.gossip_usage(switcher, "mm", pull=True, push=True)
        assert "no usage gossip" in capsys.readouterr().out
