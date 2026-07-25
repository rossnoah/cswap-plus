"""slot_order: the LWW stamp that syncs account list order across devices."""

import json
from unittest.mock import patch

from claude_swap import slot_order


class TestRecordAndLoad:
    def test_roundtrip(self, tmp_path):
        rec = slot_order.record_local_order(tmp_path, "dev-a")
        assert slot_order.load_order(tmp_path) == rec
        assert rec["originDeviceId"] == "dev-a"
        assert rec["adoptedFrom"] is None

    def test_missing_and_corrupt_are_none(self, tmp_path):
        assert slot_order.load_order(tmp_path) is None
        slot_order.order_path(tmp_path).write_text("{nope")
        assert slot_order.load_order(tmp_path) is None

    def test_monotonic_under_backwards_clock(self, tmp_path):
        first = slot_order.record_local_order(tmp_path, "dev-a")
        with patch("claude_swap.slot_order.time.time",
                   return_value=first["ts"] - 3600):
            second = slot_order.record_local_order(tmp_path, "dev-a")
        assert second["ts"] > first["ts"]


class TestOrdering:
    def test_lww_strictly_greater(self):
        older = {"ts": 1.0, "originDeviceId": "a"}
        newer = {"ts": 2.0, "originDeviceId": "a"}
        assert slot_order.is_newer(newer, older)
        assert not slot_order.is_newer(older, newer)
        assert not slot_order.is_newer(older, dict(older))
        assert slot_order.is_newer(older, None)

    def test_validate_rejects_junk(self):
        assert slot_order.validate_record(None) is None
        assert slot_order.validate_record({"ts": "soon", "originDeviceId": "a"}) is None
        assert slot_order.validate_record({"ts": 1.0, "originDeviceId": ""}) is None
        assert slot_order.validate_record({"ts": True, "originDeviceId": "a"}) is None


class TestAdopt:
    def test_adopt_verbatim_with_source(self, tmp_path):
        rec = {"ts": 50.0, "originDeviceId": "peer", "adoptedFrom": None}
        assert slot_order.adopt_record(tmp_path, rec, source="mm") is True
        stored = slot_order.load_order(tmp_path)
        assert stored["ts"] == 50.0
        assert stored["adoptedFrom"] == "mm"

    def test_adopt_rechecks_under_lock(self, tmp_path):
        newer = slot_order.record_local_order(tmp_path, "dev-a")
        stale = {"ts": newer["ts"] - 1, "originDeviceId": "peer"}
        assert slot_order.adopt_record(tmp_path, stale, source="mm") is False
        assert slot_order.load_order(tmp_path) == newer
