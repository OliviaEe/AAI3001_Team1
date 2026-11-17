#!/usr/bin/env bash
# Runs ON the Google Cloud VM (Ubuntu 24.04). Started by `vm.ps1 setup`; safe to re-run.
# Installs the app, runs it as a service on 127.0.0.1:8501 (not reachable from the
# internet directly), and starts a Cloudflare quick tunnel that gives an HTTPS link.
# Live video goes through Cloudflare TURN (keys in .env), so no inbound ports are opened.
set -euo pipefail

APP_DIR="$HOME/app"
cd "$APP_DIR"

# GPU VM (vm.ps1 -Gpu): any NVIDIA PCI device (vendor id 0x10de).
GPU=0
if grep -qs 0x10de /sys/bus/pci/devices/*/vendor; then GPU=1; fi
# CPU threads for the models (app default is 2, sized for laptops). Half the vCPUs:
# the detectors and the depth model run at the same time, each with this many threads.
# Measured live on c3-highcpu-8: 2 threads ~140-190 ms/frame, 4 ~100-130, 6 ~115-135.
# `MODEL_THREADS=...` in .env overrides.
CPUS="$(nproc)"
THREADS=$(( CPUS / 2 > 2 ? CPUS / 2 : 2 ))
echo "Mode: $([ "$GPU" = 1 ] && echo GPU || echo CPU) | $CPUS vCPUs | MODEL_THREADS=$THREADS"

echo "== System packages (OpenCV needs libGL and GLib) =="
sudo apt-get update -y
sudo apt-get install -y libgl1 libglib2.0-0t64 curl ca-certificates

if [ "$GPU" = 1 ]; then
    echo "== NVIDIA driver =="
    if ! nvidia-smi >/dev/null 2>&1; then
        sudo apt-get install -y ubuntu-drivers-common
        # Newest data-centre ("server") driver Ubuntu offers for this GPU and kernel.
        VER="$(ubuntu-drivers list --gpgpu 2>/dev/null | grep -o 'nvidia-driver-[0-9]*-server' \
               | grep -o '[0-9]\+' | sort -n | tail -n 1 || true)"
        if [ -z "$VER" ]; then echo "ERROR: no NVIDIA server driver found for this GPU."; exit 1; fi
        sudo ubuntu-drivers install --gpgpu "nvidia:${VER}-server"
        sudo apt-get install -y "nvidia-utils-${VER}-server"
        sudo modprobe nvidia 2>/dev/null || true      # may need a reboot instead (handled below)
    fi
fi

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
PY=.venv/bin/python
if [ "$GPU" = 1 ]; then
    # PyPI's Linux PyTorch build includes CUDA. Replace a CPU-only build if one is installed.
    if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.version.cuda else 1)' 2>/dev/null; then
        "$UV" pip install --python "$PY" --reinstall-package torch --reinstall-package torchvision \
            torch==2.12.0 torchvision==0.27.0
    fi
else
    # CPU-only PyTorch: the default Linux build adds several GB of unused CUDA libraries.
    "$UV" pip install --python "$PY" torch==2.12.0 torchvision==0.27.0 \
        --index-url https://download.pytorch.org/whl/cpu
fi
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
Environment=MODEL_THREADS=$THREADS
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

if [ "$GPU" = 1 ]; then
    echo "== GPU check =="
    if nvidia-smi -L >/dev/null 2>&1 && "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
        echo "GPU ready: $(nvidia-smi -L | head -n 1)"
    else
        # A fresh driver often loads only after a reboot. vm.ps1 sees this line, restarts
        # the VM and prints the link; the services start by themselves on boot.
        echo "REBOOT_NEEDED"
        exit 0
    fi
fi

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
