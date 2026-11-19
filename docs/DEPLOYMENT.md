# Cloud Deployment

[Project overview](../README.md) · [UI guide](UI_GUIDE.md) · [Model results](MODEL_RESULTS.md)

**Streamlit Community Cloud is the main deployment. Google Cloud is the backup**, for when Streamlit Community Cloud is unavailable or too slow. Both use the same **Cloudflare TURN** relay for the live camera.

| | Streamlit Community Cloud (main) | Google Cloud (backup) |
|:---|:---|:---|
| Cost | Free | Pay per hour while running (stop it after use) |
| Hardware | Shared, small | Dedicated 8 vCPU server, or an NVIDIA T4 GPU server |
| Link | Fixed `*.streamlit.app` address | New `*.trycloudflare.com` address on every start |
| Setup | Deploys from GitHub | Scripts in [`deploy/gcp/`](../deploy/gcp/) |
| Camera relay | Cloudflare TURN (keys in Streamlit Secrets) | Cloudflare TURN (keys in `.env`) |

### Streamlit Community Cloud (main)

The Child Safety Monitor is deployed on **Streamlit Community Cloud** through Olivia's GitHub fork.

**Live application:** [Child Safety Monitor](https://aai3001team1-cvdl-child-safety-monitor.streamlit.app)

The deployment runs `app.py` using Python 3.13. Python dependencies are installed from `requirements.txt`, while `packages.txt` supplies the Linux libraries required by OpenCV:

```text
libgl1
libglib2.0-0t64
```

The Telegram bot token and Cloudflare TURN credentials are configured through Streamlit's Secrets settings. Cloudflare TURN provides a video relay when a direct camera connection cannot be established. The Cloudflare credentials remain in Secrets.

For remote camera connection issues, see the
[cloud camera setup](UI_GUIDE.md#cloud-camera-setup).

Cloud testing confirmed that uploaded-image inference, live camera monitoring, and Telegram alert delivery work. A teammate who previously could not connect their camera successfully tested live monitoring after Cloudflare TURN was configured. Processing speed depends on the available cloud resources and network connection. TURN improves connectivity but does not increase the app's CPU resources. Local database storage is not guaranteed to persist across cloud rebuilds or restarts.

### Google Cloud (backup)

If Streamlit Community Cloud is down, asleep or too slow for a demo, the same `app.py` can run on a Google Cloud Compute Engine virtual machine (VM). It provides dedicated resources and incurs charges while running. The performance table below records the team's CPU-server measurements; results depend on workload and hardware. One PowerShell script on your PC, [`deploy/gcp/vm.ps1`](../deploy/gcp/vm.ps1), creates, updates, starts and stops it. The VM opens **no inbound ports**:

```text
Browser --HTTPS-->  Cloudflare quick tunnel  --> VM: Streamlit on 127.0.0.1:8501 (systemd service)
Browser <--video--> Cloudflare TURN relay   <--> VM: streamlit-webrtc (aiortc)
```

- **Web page:** a [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) gives the app a free HTTPS link. Browsers only allow camera access over HTTPS.
- **Live video:** WebRTC tries a direct connection first and falls back to the [Cloudflare TURN relay](https://developers.cloudflare.com/realtime/turn/), the same relay used by Streamlit Community Cloud. The relay lets the camera work on mobile data and school or office Wi-Fi, which often block direct connections.
- **App service:** the app runs as a `systemd` service, so it starts on boot and restarts if it crashes.

#### Servers

| Option | Machine | Approx. cost while running | Notes |
|:---|:---|:---|:---|
| CPU (default) | `c3-highcpu-8`: 8 vCPU, 16 GB RAM | US$0.35–0.40 / hour | Used for testing so far |
| GPU (`-Gpu`) | `n1-standard-4` + NVIDIA T4: 4 vCPU, 15 GB RAM | ≈ US$0.60 / hour | Needs GPU quota first; not yet tested |

Both use zone `asia-southeast1-b` (Singapore). A stopped VM costs nothing for compute; its 30–40 GB disk costs a few dollars a month. The prices above are recorded estimates rather than current quotes; check the [Google Cloud pricing calculator](https://cloud.google.com/products/calculator).

#### Prerequisites (once)

1. **Google Cloud project with billing.** Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), run `gcloud init`, then enable Compute Engine:
   ```powershell
   gcloud services enable compute.googleapis.com
   ```
   Add a budget alert under **Billing → Budgets & alerts**. Budgets only send emails; they do not stop spending.
2. **Cloudflare TURN key.** Use the same TURN key ID and API token as the Streamlit Community Cloud Secrets, or create a key in the Cloudflare dashboard under **Realtime → TURN Server** (check Cloudflare's current usage allowance and pricing before deployment).
3. **`.env` file.** Copy [`.example.env`](../.example.env) to `.env` and fill in the values below. `.env` is ignored by Git. Never commit keys or tokens.
4. **GPU only.** Request GPU quota: **IAM & Admin → Quotas**, search "GPUs (all regions)", request **1**.

| `.env` setting | Required | Purpose |
|:---|:---:|:---|
| `CLOUDFLARE_TURN_KEY_ID`, `CLOUDFLARE_TURN_KEY_API_TOKEN` | Yes | Camera relay (TURN) |
| `TELEGRAM_BOT_TOKEN` | No | Caregiver alerts. Without it, alerts are off |
| `MODEL_THREADS` | No | CPU threads for the models. Leave unset: laptops use 2, the setup script uses half the VM's vCPUs |

#### First deployment

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\gcp\vm.ps1 create   # create the VM
powershell -ExecutionPolicy Bypass -File deploy\gcp\vm.ps1 upload   # copy app, weights and .env
powershell -ExecutionPolicy Bypass -File deploy\gcp\vm.ps1 setup    # install and start; prints the link
```

`upload` copies files directly from your PC, so you don't need to push to GitHub first. `setup` takes about 5–10 minutes the first time. It:
- installs Python 3.13 and the dependencies (CPU-only PyTorch, or the CUDA build plus the NVIDIA driver on the GPU VM);
- starts the app and tunnel services;
- prints the `https://….trycloudflare.com` link.

Add `-Gpu` to every command to manage the GPU server instead. Its first `setup` may restart the VM once to load the NVIDIA driver; the script handles this.

#### Everyday use

| Task | Command (`powershell -ExecutionPolicy Bypass -File deploy\gcp\vm.ps1 …`) |
|:---|:---|
| Start the server and print the new link (≈ 1 minute) | `start` |
| **Stop the server after every demo or test** | `stop` |
| Show the current link again | `url` |
| Deploy code changes | `upload`, then `setup` |
| Show recent app logs | `logs` |
| Open a shell on the VM | `ssh` |
| Delete the VM and its disk | `delete` |

- **The link changes every time the server starts.** Use the link printed by the most recent `start` or `setup`. An old link shows a Cloudflare tunnel error (530).
- **The first page load after a start takes about a minute** while the models load.
- In the Cloud Console (**Compute Engine → VM instances**) you can also start or stop the VM, but run `url` afterwards to get the link.

#### Performance (CPU server)

Team-recorded timings from the live app (**Performance details → Latency**). These measurements describe the processing shown by that panel, not camera-to-notification latency or a guaranteed frame rate:

| `MODEL_THREADS` | Latency per frame |
|:---:|:---:|
| 2 (laptop default) | 140–190 ms |
| **4 (VM default)** | **100–130 ms** |
| 6 | 115–135 ms |

More threads is not always faster: the detectors and the depth model run at the same time and compete for cores. At about 120 ms per frame, **Settings → Max model runs / second** can be raised from 5 to about 8.

#### Notes

- **One Telegram poller per bot token.** Do not run the Google Cloud VM, Streamlit Community Cloud and a local copy with the same `TELEGRAM_BOT_TOKEN` at the same time, or bot commands will conflict.
- **Subscriptions persist** on the VM's disk (`subscribers.db`) across stop and start, unlike Streamlit Community Cloud. They are lost if the VM is deleted.
- **TURN credentials reach the browser.** They are short-lived (at most about 2 hours). Deleting the TURN key in Cloudflare disables the relay immediately.
