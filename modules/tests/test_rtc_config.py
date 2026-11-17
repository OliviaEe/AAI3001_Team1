"""Connection setup tests use stand-in providers, never real TURN accounts."""
import pytest

from modules.rtc_config import camera_connection_label, camera_rtc_configuration, fetch_metered_ice_servers


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


METERED_KEYS = {"METERED_TURN_DOMAIN": "example.metered.live", "METERED_TURN_API_KEY": "metered-test-key"}


def test_metered_takes_precedence_and_keeps_api_key_out_of_rtc_settings():
    def provider(**kwargs):
        assert kwargs == {"domain": "example.metered.live", "api_key": "metered-test-key"}
        return [{"urls": "turn:standard.relay.metered.ca:80", "username": "user", "credential": "password"}]

    config, method, warning = camera_rtc_configuration(
        APP_URL, (KEYS | METERED_KEYS).get, unused_provider, provider,
    )
    assert method == "Metered TURN available"
    assert "metered-test-key" not in repr(config)
    assert warning is None


def test_metered_failure_does_not_fall_back_to_cloudflare_or_expose_api_key():
    def provider(**kwargs):
        raise RuntimeError("Failed URL contains metered-test-key")

    config, method, warning = camera_rtc_configuration(
        APP_URL, (KEYS | METERED_KEYS).get, unused_provider, provider,
    )
    assert "STUN only" in method
    assert "Metered" in warning and "unavailable" in warning
    assert "metered-test-key" not in repr((config, method, warning))


def test_partial_metered_configuration_does_not_activate_cloudflare():
    config, method, warning = camera_rtc_configuration(
        APP_URL, (KEYS | {"METERED_TURN_DOMAIN": "example.metered.live"}).get,
        unused_provider, unused_provider,
    )
    assert "incomplete" in warning and "Metered" in warning


@pytest.mark.parametrize("domain", ["example.metered.live", "https://example.metered.live/"])
def test_metered_request_uses_credential_endpoint(domain):
    from types import SimpleNamespace

    def get(url, **kwargs):
        assert url == "https://example.metered.live/api/v1/turn/credentials"
        assert kwargs == {"params": {"apiKey": "test-key"}, "timeout": 10}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [{"urls": "turn:test:80"}])

    assert fetch_metered_ice_servers(domain, "test-key", http_get=get) == [{"urls": "turn:test:80"}]


@pytest.mark.parametrize("domain", [
    "http://example.metered.live", "https://example.com", "https://example.metered.live.evil.com",
    "https://user:password@example.metered.live", "https://example.metered.live/custom",
])
def test_metered_domain_cannot_send_api_key_to_another_service(domain):
    with pytest.raises(ValueError):
        fetch_metered_ice_servers(domain, "test-key", http_get=unused_provider)
