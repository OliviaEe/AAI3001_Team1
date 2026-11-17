#!/usr/bin/env bash
# Runs ON the Google Cloud VM (Ubuntu 24.04). Started by `vm.ps1 setup`; safe to re-run.
# Installs the app, runs it as a service on 127.0.0.1:8501 (not reachable from the
# internet directly), and starts a Cloudflare quick tunnel that gives an HTTPS link.
# Live video goes through Cloudflare TURN (keys in .env), so no inbound ports are opened.
set -euo pipefail

APP_DIR="$HOME/app"
cd "$APP_DIR"

echo "== System packages (OpenCV needs libGL and GLib) =="
sudo apt-get update -y
sudo apt-get install -y libgl1 libglib2.0-0t64 curl ca-certificates

echo "== Secrets =="
mkdir -p .streamlit
if [ -f "$HOME/secrets.toml" ]; then mv "$HOME/secrets.toml" .streamlit/secrets.toml; fi
if [ -f "$HOME/app.env" ]; then mv "$HOME/app.env" .env; fi
[ -f .env ] && sed -i 's/\r$//' .env && chmod 600 .env
[ -f .streamlit/secrets.toml ] && chmod 600 .streamlit/secrets.toml
[ -f .env ] || echo "WARNING: no .env -> no Cloudflare TURN; the camera may fail on strict networks."

echo "== Python 3.13 environment =="
if [ ! -x "$HOME/.local/bin/uv" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
UV="$HOME/.local/bin/uv"
"$UV" venv --python 3.13 --allow-existing .venv
# CPU-only PyTorch first: the default Linux build adds several GB of unused CUDA libraries.
"$UV" pip install --python .venv/bin/python torch==2.12.0 torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cpu
grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/requirements-no-torch.txt
"$UV" pip install --python .venv/bin/python -r /tmp/requirements-no-torch.txt

echo "== cloudflared =="
if ! command -v cloudflared >/dev/null; then
    curl -fsSL -o /tmp/cloudflared.deb \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i /tmp/cloudflared.deb
fi
CLOUDFLARED="$(command -v cloudflared)"

echo "== Services =="
sudo tee /etc/systemd/system/child-safety.service >/dev/null <<EOF
[Unit]
Description=Child Safety Monitor (Streamlit)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=-$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/child-safety-tunnel.service >/dev/null <<EOF
[Unit]
Description=Cloudflare quick tunnel (HTTPS link) for Child Safety Monitor
After=network-online.target child-safety.service
Wants=network-online.target

[Service]
User=$USER
ExecStart=$CLOUDFLARED tunnel --no-autoupdate --url http://127.0.0.1:8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable child-safety child-safety-tunnel
# Restarting the tunnel ends the old link, so only read links logged after this moment
# (the log still holds the previous link from earlier in this boot).
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
sudo systemctl restart child-safety child-safety-tunnel

echo "== Waiting for the new HTTPS link =="
for _ in $(seq 1 30); do
    url="$(journalctl -u child-safety-tunnel --since "$SINCE" -o cat --no-pager | grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' | tail -n 1 || true)"
    if [ -n "$url" ]; then echo "App link: $url"; exit 0; fi
    sleep 2
done
echo "Tunnel link not found yet. Run 'vm.ps1 url' in a minute."
