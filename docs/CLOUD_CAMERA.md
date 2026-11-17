# Cloud camera connections

The cloud app can use Metered or Cloudflare TURN when direct WebRTC connections
cannot reach a viewer's network. TURN does not fix camera permissions, a
disconnected Streamlit session, or CPU throttling.

## Metered: no-card connection test

Metered's [pricing page](https://www.metered.ca/pricing) currently lists a free
500 MB monthly TURN trial with no credit card required. Metered counts both
ingress and egress. Check the allowance shown in your account and keep tests short.

1. Create an account at <https://dashboard.metered.ca/signup> and choose the free TURN plan.
2. Create an app and open **TURN Server > Credentials**. Generate a TURN credential
   using the [credential guide](https://www.metered.ca/docs/turn-server-service/creating-turn-credentials/).
3. Open the generated credential and copy its **username** and **password** privately.
4. Add these top-level entries in **Streamlit App settings > Secrets**, alongside
   the existing Telegram secret. Replace the placeholders privately:

   ```toml
   METERED_TURN_USERNAME = "your-turn-username"
   METERED_TURN_PASSWORD = "your-turn-password"
   ```

5. Allow up to two minutes for a newly created credential to propagate, then save
   the settings and reboot the app. Open a fresh app tab after startup.
6. Under **Performance details**, confirm **Metered TURN configured**, then retry
   the camera on the affected device. Stop monitoring after the test and check usage.

The direct setup uses the `global.relay.metered.ca` hostname shown in the
dashboard snippet, including TLS on port 443. If your snippet shows another
Metered relay host, set `METERED_TURN_HOST` to that hostname without a scheme or
port. The configured username and password are passed to WebRTC for TURN
authentication. The label means configuration is present; it does not verify
the password or a successful relay allocation. Replace credentials when expired
or revoked.

### Optional credential API setup

If your dashboard also exposes a credential-scoped API key, you can use it
instead of the username/password entries:

```toml
METERED_TURN_DOMAIN = "your-app.metered.live"
METERED_TURN_API_KEY = "your-credential-api-key"
```

The server retrieves ICE credentials using Metered's
[credential endpoint](https://www.metered.ca/docs/turn-rest-api/get-credential/)
and caches the result for 60 seconds. The API key is not included in WebRTC settings.
The lifetime of returned TURN credentials is controlled by the credential created
in Metered. Replace the credential's API key in Secrets after it expires or is revoked.

## Cloudflare: optional alternative

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
ICE credentials. Only the expiring credentials are passed to WebRTC.

Cloudflare usage can incur charges beyond its allowance. Check the
[current pricing](https://developers.cloudflare.com/realtime/sfu/platform/pricing/)
and your account's terms before activating the service. The app does not create
an account or enable billing.

## Provider selection and connection status

Metered username/password entries take precedence over its API setup, which
takes precedence over Cloudflare. Incomplete
Metered settings or a failed Metered request show a warning and use STUN; they
never silently activate Cloudflare. Without Metered entries, Cloudflare is used
only when both Cloudflare entries are configured. Without either provider, the
app retains Google STUN. Localhost continues to use local connections.

Provider errors show fixed messages without API key values. A reported TURN
configuration does not prove that the selected connection uses the relay.
Confirm video and analysed frames on the affected device before marking an issue fixed.
