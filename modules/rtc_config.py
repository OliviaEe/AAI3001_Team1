"""Camera connection configuration without exposing long-lived TURN keys."""
import re
from urllib.parse import urlsplit


def fetch_metered_ice_servers(domain, api_key, *, http_get=None):
    """Read ICE servers using a credential-scoped Metered API key."""
    value = domain.strip()
    parsed = urlsplit(value if "://" in value else "https://" + value)
    if (parsed.scheme != "https" or parsed.netloc != parsed.hostname
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.metered\.live", parsed.hostname or "")):
        raise ValueError("Use your app's Metered domain, such as example.metered.live")
    if http_get is None:
        import requests
        http_get = requests.get
    response = http_get(
        f"https://{parsed.hostname}/api/v1/turn/credentials",
        params={"apiKey": api_key}, timeout=10,
    )
    response.raise_for_status()
    return response.json()


def camera_rtc_configuration(app_url, read_setting, fetch_turn_servers, fetch_metered_servers=None):
    """Return (RTC settings, connection method, setup warning).

    Provider API keys stay on the server. Only ICE credentials go to WebRTC.
    Metered settings take precedence; a failure never enables another provider.
    """
    if urlsplit(app_url).hostname in {"localhost", "127.0.0.1", "::1"}:
        return {"iceServers": []}, "Local connection", None

    direct = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    domain = read_setting("METERED_TURN_DOMAIN")
    metered_key = read_setting("METERED_TURN_API_KEY")
    if domain or metered_key:
        provider = "Metered"
        if not domain or not metered_key:
            return direct, "Direct connection (STUN only)", (
                "Metered camera relay setup is incomplete. Add both METERED_TURN_DOMAIN "
                "and METERED_TURN_API_KEY in Streamlit Secrets."
            )
        fetch = fetch_metered_servers or fetch_metered_ice_servers
        kwargs = {"domain": domain, "api_key": metered_key}
    else:
        provider = "Cloudflare"
        key_id = read_setting("CLOUDFLARE_TURN_KEY_ID")
        api_token = read_setting("CLOUDFLARE_TURN_KEY_API_TOKEN")
        if not key_id and not api_token:
            return direct, "Direct connection (STUN only)", None
        if not key_id or not api_token:
            return direct, "Direct connection (STUN only)", (
                "Cloudflare camera relay setup is incomplete. Add both CLOUDFLARE_TURN_KEY_ID "
                "and CLOUDFLARE_TURN_KEY_API_TOKEN in Streamlit Secrets."
            )
        fetch = fetch_turn_servers
        kwargs = {"turn_key_id": key_id, "turn_key_api_token": api_token}

    try:
        servers = fetch(**kwargs)
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
            f"{provider} camera relay is unavailable. Check the TURN keys and "
            "service settings. The app is trying a direct connection instead."
        )
    return {"iceServers": servers}, f"{provider} TURN available", None


def camera_connection_label(requested, playing):
    if playing:
        return "Camera on" if requested else "Stopping camera"
    return "Connecting" if requested else "Camera stopped"
