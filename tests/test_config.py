"""Settings tests.

Mostly one bug, caught once and now guarded permanently: `Settings` is a slots
dataclass, so a class attribute is a member descriptor rather than the default
value. `from_env` used those as its os.getenv fallbacks, and five settings
silently became descriptors whenever the environment variable was unset —
which is the normal case. Nothing failed until the engine read them.
"""

import os
from unittest import mock

from researchlens.config import Settings


def test_every_default_is_a_real_value_not_a_slot_descriptor():
    with mock.patch.dict(os.environ, {}, clear=True):
        s = Settings.from_env()
    for field in (
        "mode", "embedding_model", "reranker_model", "ollama_host",
        "ollama_model", "hosted_model",
    ):
        value = getattr(s, field)
        assert isinstance(value, str), f"{field} is {type(value).__name__}"
        assert value and "member" not in value.lower()
    assert isinstance(s.allowed_origins, tuple) and s.allowed_origins


def test_the_environment_overrides_a_default():
    with mock.patch.dict(os.environ, {"OLLAMA_MODEL": "llama3.2:1b"}, clear=True):
        assert Settings.from_env().ollama_model == "llama3.2:1b"


def test_no_hosted_key_is_a_supported_state():
    """The moment a key is required, the project's central claim stops being
    true — so an absent one must not be an error."""
    with mock.patch.dict(os.environ, {}, clear=True):
        s = Settings.from_env()
    assert s.hosted_available is False
    assert s.uploads_enabled is True


def test_demo_mode_disables_uploads():
    with mock.patch.dict(os.environ, {"RESEARCHLENS_MODE": "demo"}, clear=True):
        s = Settings.from_env()
    assert s.mode == "demo"
    assert s.uploads_enabled is False


def test_an_unknown_mode_is_rejected_at_startup():
    with mock.patch.dict(os.environ, {"RESEARCHLENS_MODE": "production"}, clear=True):
        try:
            Settings.from_env()
        except ValueError as e:
            assert "local" in str(e) and "demo" in str(e)
        else:
            raise AssertionError("an unknown mode should not be accepted")


def test_allowed_origins_can_be_set_from_the_environment():
    with mock.patch.dict(
        os.environ, {"ALLOWED_ORIGINS": "https://a.example, https://b.example"}, clear=True
    ):
        assert Settings.from_env().allowed_origins == ("https://a.example", "https://b.example")


def test_providers_default_to_both():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert Settings.from_env().providers == ("local", "hosted")


def test_an_instance_can_offer_only_hosted():
    """A Space has no persistent volume and therefore no local model. Offering
    one would put a choice on the page that can never answer."""
    with mock.patch.dict(os.environ, {"PROVIDERS": "hosted"}, clear=True):
        assert Settings.from_env().providers == ("hosted",)
