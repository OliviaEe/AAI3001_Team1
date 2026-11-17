# Cloud camera connections

The cloud app can use Cloudflare Realtime TURN when direct WebRTC connections
cannot reach a viewer's network. TURN is a connection option; it does not fix
camera permissions or a disconnected Streamlit session.

## Configure the relay

1. Create a TURN key in your Cloudflare account using the
   [Cloudflare credential guide](https://developers.cloudflare.com/realtime/turn/generate-credentials/).
2. Add the following top-level entries in Streamlit Community Cloud's **App settings > Secrets**,
   alongside the existing Telegram secret. Replace the placeholders privately:

   ```toml
   CLOUDFLARE_TURN_KEY_ID = "your-turn-key-id"
   CLOUDFLARE_TURN_KEY_API_TOKEN = "your-turn-key-api-token"
   ```

3. Save the settings and reboot the app. Open a fresh app tab after startup.
4. Under **Performance details**, confirm that the connection method says
   **Cloudflare TURN available**, then retry the camera on the affected device.

The app uses the camera library to exchange the server-side keys for expiring
ICE credentials. Only the expiring credentials are passed to WebRTC. A reported
TURN configuration does not prove that the selected connection uses the relay.
Confirm video and analysed frames on the affected device before marking the issue fixed.

Without keys, the app keeps its existing Google STUN configuration. Localhost
continues to use local connections. Partial credentials or provider failures
show a setup warning and fall back to STUN; no key values are displayed.

Cloudflare usage can incur charges beyond its allowance. Check the
[current pricing](https://developers.cloudflare.com/realtime/sfu/platform/pricing/)
and your account's terms before activating the service. The app does not create
an account or enable billing.
