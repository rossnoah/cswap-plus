"""active_intent: the LWW record that syncs the active account across devices."""

import json
from unittest.mock import patch

from claude_swap import active_intent


def _record(tmp_path, email="a@x.com", device="dev-a", kind="manual"):
    return active_intent.record_local_intent(
        tmp_path, email=email, org_uuid="", device_id=device, kind=kind
    )


class TestRecordAndLoad:
    def test_roundtrip(self, tmp_path):
        intent = _record(tmp_path)
        loaded = active_intent.load_intent(tmp_path)
        assert loaded == intent
        assert loaded["email"] == "a@x.com"
        assert loaded["originKind"] == "manual"
        assert loaded["adoptedFrom"] is None

    def test_missing_file_is_none(self, tmp_path):
        assert active_intent.load_intent(tmp_path) is None

    def test_corrupt_file_is_none(self, tmp_path):
        active_intent.intent_path(tmp_path).write_text("{nope")
        assert active_intent.load_intent(tmp_path) is None

    def test_wrong_schema_version_is_none(self, tmp_path):
        active_intent.intent_path(tmp_path).write_text(
            json.dumps({"schemaVersion": 99, "intent": {}})
        )
        assert active_intent.load_intent(tmp_path) is None

    def test_monotonic_ts_survives_backwards_clock(self, tmp_path):
        first = _record(tmp_path)
        # Clock steps backwards: the next local record must still be newer.
        with patch("claude_swap.active_intent.time.time",
                   return_value=first["ts"] - 3600):
            second = _record(tmp_path, email="b@x.com")
        assert second["ts"] > first["ts"]
        assert active_intent.is_newer(second, first)


class TestOrdering:
    def test_newer_ts_wins(self):
        older = {"ts": 1.0, "originDeviceId": "a"}
        newer = {"ts": 2.0, "originDeviceId": "a"}
        assert active_intent.is_newer(newer, older)
        assert not active_intent.is_newer(older, newer)

    def test_tie_loses(self):
        a = {"ts": 1.0, "originDeviceId": "a"}
        assert not active_intent.is_newer(a, dict(a))

    def test_device_id_breaks_ts_tie(self):
        a = {"ts": 1.0, "originDeviceId": "aaa"}
        b = {"ts": 1.0, "originDeviceId": "bbb"}
        assert active_intent.is_newer(b, a)
        assert not active_intent.is_newer(a, b)

    def test_none_baseline_always_loses(self):
        assert active_intent.is_newer({"ts": 0.0, "originDeviceId": ""}, None)

    def test_malformed_baseline_loses(self):
        assert active_intent.intent_key({"ts": "soon"}) == (float("-inf"), "")


class TestValidate:
    def test_valid_payload(self):
        intent = {
            "email": "a@x.com", "organizationUuid": "", "ts": 5.0,
            "originDeviceId": "d", "originKind": "manual",
        }
        payload = {"schemaVersion": 1, "intent": intent}
        out = active_intent.validate_intent(payload)
        assert out["email"] == "a@x.com"
        assert out["ts"] == 5.0

    def test_rejects_bad_shapes(self):
        assert active_intent.validate_intent(None) is None
        assert active_intent.validate_intent({"schemaVersion": 1}) is None
        assert active_intent.validate_intent(
            {"schemaVersion": 2, "intent": {}}
        ) is None
        for broken in (
            {"email": "", "ts": 1.0, "originDeviceId": "d", "originKind": "manual"},
            {"email": "a@x", "ts": True, "originDeviceId": "d", "originKind": "manual"},
            {"email": "a@x", "ts": 1.0, "originDeviceId": "", "originKind": "manual"},
            {"email": "a@x", "ts": 1.0, "originDeviceId": "d", "originKind": "cosmic"},
        ):
            assert active_intent.validate_intent(
                {"schemaVersion": 1, "intent": broken}
            ) is None, broken

    def test_org_uuid_coerced_to_empty(self):
        out = active_intent.validate_intent_record(
            {"email": "a@x", "ts": 1.0, "originDeviceId": "d",
             "originKind": "auto", "organizationUuid": None}
        )
        assert out["organizationUuid"] == ""


class TestAdopt:
    def test_adopt_stores_verbatim_with_source(self, tmp_path):
        intent = {
            "email": "b@x.com", "organizationUuid": "", "ts": 100.0,
            "originDeviceId": "peer-dev", "originKind": "manual",
            "adoptedFrom": None,
        }
        assert active_intent.adopt_intent(tmp_path, intent, source="mm") is True
        loaded = active_intent.load_intent(tmp_path)
        assert loaded["ts"] == 100.0
        assert loaded["originDeviceId"] == "peer-dev"
        assert loaded["adoptedFrom"] == "mm"

    def test_adopt_rechecks_freshness_under_lock(self, tmp_path):
        newer = _record(tmp_path)
        stale = {
            "email": "b@x.com", "organizationUuid": "",
            "ts": newer["ts"] - 10, "originDeviceId": "peer",
            "originKind": "manual",
        }
        assert active_intent.adopt_intent(tmp_path, stale, source="mm") is False
        assert active_intent.load_intent(tmp_path) == newer

    def test_adopt_rejects_malformed(self, tmp_path):
        assert active_intent.adopt_intent(tmp_path, {"ts": 1.0}, source="mm") is False
        assert active_intent.load_intent(tmp_path) is None
