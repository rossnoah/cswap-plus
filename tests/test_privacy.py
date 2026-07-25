"""Masking helpers and the ui.privacy setting."""

import json

from claude_swap.privacy import mask_email, mask_org, mask_text
from claude_swap.settings import load_ui_settings


class TestMaskEmail:
    def test_masks_local_and_domain_to_first_char(self):
        assert mask_email("noahwross@gmail.com") == "n•••@g•••"

    def test_distinct_first_chars_stay_distinguishable(self):
        assert mask_email("alice@acme.dev") != mask_email("bob@acme.dev")

    def test_empty_string_passes_through(self):
        assert mask_email("") == ""

    def test_no_at_sign_still_masked(self):
        assert mask_email("not-an-email") == "n•••"


class TestMaskOrg:
    def test_masks_to_first_char(self):
        assert mask_org("Anthropic") == "•••"

    def test_personal_tag_passes_through(self):
        assert mask_org("personal") == "personal"

    def test_empty_passes_through(self):
        assert mask_org("") == ""


class TestMaskText:
    def test_redacts_every_email_in_free_text(self):
        out = mask_text("Added Account 9: fresh@example.com (was old@other.org)")
        assert "fresh@example.com" not in out
        assert "old@other.org" not in out
        assert "f•••@e•••" in out
        assert "o•••@o•••" in out

    def test_text_without_emails_unchanged(self):
        assert mask_text("Switched to account 3") == "Switched to account 3"


class TestPrivacySetting:
    def test_defaults_off(self, tmp_path):
        assert load_ui_settings(tmp_path).privacy is False

    def test_loads_true(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            json.dumps({"ui": {"privacy": True}})
        )
        assert load_ui_settings(tmp_path).privacy is True

    def test_non_bool_falls_back_to_default(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            json.dumps({"ui": {"privacy": "yes"}})
        )
        assert load_ui_settings(tmp_path).privacy is False

    def test_bad_privacy_does_not_discard_theme(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            json.dumps({"ui": {"theme": "light", "privacy": 3}})
        )
        ui = load_ui_settings(tmp_path)
        assert ui.theme == "light"
        assert ui.privacy is False


class TestSlotAwareMask:
    def test_known_slot_renders_account_n(self):
        assert mask_email("noahwross@gmail.com", 2) == "Account 2"
        assert mask_email("noahwross@gmail.com", "5") == "Account 5"

    def test_no_slot_keeps_char_mask(self):
        assert mask_email("noahwross@gmail.com") == "n•••@g•••"
