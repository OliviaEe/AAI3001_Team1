"""Connection setup tests use stand-in providers, never real TURN accounts."""
import pytest

from modules.rtc_config import camera_connection_label, camera_rtc_configuration


KEYS = {"CLOUDFLARE_TURN_KEY_ID": "test-key", "CLOUDFLARE_TURN_KEY_API_TOKEN": "test-secret"}
APP_URL = "https://example.streamlit.app"


def unused_provider(**kwargs):
    raise AssertionError("Provider must not be called")


@pytest.mark.parametrize("url", ["http://localhost:8501", "http://127.0.0.1:8501", "http://[::1]:8501"])
def test_local_camera_does_not_contact_turn_provider(url):
    config, method, warning = camera_rtc_configuration(url, KEYS.get, unused_provider)
    assert config == {"iceServers": []}
    assert warning is None


def test_cloud_without_keys_preserves_stun_setup():
    config, method, warning = camera_rtc_configuration(APP_URL, {}.get, unused_provider)
    assert config["iceServers"][0]["urls"] == ["stun:stun.l.google.com:19302"]
    assert warning is None


def test_incomplete_keys_show_warning_without_contacting_provider():
    config, method, warning = camera_rtc_configuration(
        APP_URL, {"CLOUDFLARE_TURN_KEY_ID": "test-key"}.get, unused_provider,
    )
    assert "incomplete" in warning


def test_short_lived_credentials_are_passed_to_webrtc_without_long_lived_keys():
    servers = [{"urls": ["turns:turn.example.com:443?transport=tcp"],
                "username": "temporary-user", "credential": "temporary-password"}]

    def provider(**kwargs):
        assert kwargs == {"turn_key_id": "test-key", "turn_key_api_token": "test-secret"}
        return servers

    config, method, warning = camera_rtc_configuration(APP_URL, KEYS.get, provider)
    assert config == {"iceServers": servers}
    assert "test-secret" not in repr(config)
    assert "Cloudflare TURN" in method
    assert warning is None


def test_provider_error_does_not_leak_credentials_or_crash_camera_setup():
    def provider(**kwargs):
        raise RuntimeError("Do not expose test-secret")

    config, method, warning = camera_rtc_configuration(APP_URL, KEYS.get, provider)
    assert "stun:" in repr(config)
    assert "unavailable" in warning
    assert "test-secret" not in repr((config, method, warning))


def test_stun_only_provider_response_is_not_reported_as_turn_available():
    config, method, warning = camera_rtc_configuration(
        APP_URL, KEYS.get, lambda **kwargs: [{"urls": "stun:example.com:3478"}],
    )
    assert "unavailable" in warning
    assert "STUN only" in method


@pytest.mark.parametrize("requested,playing,label", [
    (True, False, "Connecting"), (True, True, "Camera on"),
    (False, False, "Camera stopped"), (False, True, "Stopping camera"),
])
def test_status_reports_actual_connection(requested, playing, label):
    assert camera_connection_label(requested, playing) == label
