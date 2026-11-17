"""Camera connection configuration without exposing long-lived TURN keys."""
from urllib.parse import urlsplit


def camera_rtc_configuration(app_url, read_setting, fetch_turn_servers):
    """Return (RTC settings, connection method, setup warning).

    The provider exchanges server-side keys for short-lived ICE credentials.
    Failed setup keeps direct connections available and reports the fallback.
    """
    if urlsplit(app_url).hostname in {"localhost", "127.0.0.1", "::1"}:
        return {"iceServers": []}, "Local connection", None

    direct = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    key_id = read_setting("CLOUDFLARE_TURN_KEY_ID")
    api_token = read_setting("CLOUDFLARE_TURN_KEY_API_TOKEN")
    if not key_id and not api_token:
        return direct, "Direct connection (STUN only)", None
    if not key_id or not api_token:
        return direct, "Direct connection (STUN only)", (
            "Camera relay setup is incomplete. Add both CLOUDFLARE_TURN_KEY_ID "
            "and CLOUDFLARE_TURN_KEY_API_TOKEN in Streamlit Secrets."
        )

    try:
        servers = fetch_turn_servers(turn_key_id=key_id, turn_key_api_token=api_token)
        has_relay = False
        for server in servers:
            urls = server.get("urls", [])
            urls = [urls] if isinstance(urls, str) else urls
            has_relay |= any(url.startswith(("turn:", "turns:")) for url in urls)
        if not has_relay:
            raise ValueError("No TURN server returned")
    except Exception:
        # Provider errors can contain credentials. Show only a fixed message.
        return direct, "Direct connection (STUN only)", (
            "Cloudflare camera relay is unavailable. Check the TURN keys and "
            "service settings. The app is trying a direct connection instead."
        )
    return {"iceServers": servers}, "Cloudflare TURN available", None


def camera_connection_label(requested, playing):
    if playing:
        return "Camera on" if requested else "Stopping camera"
    return "Connecting" if requested else "Camera stopped"
