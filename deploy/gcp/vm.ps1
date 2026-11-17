<#
Google Cloud demo server for the Child Safety Monitor. Run from the repo root on your PC:

    powershell -ExecutionPolicy Bypass -File deploy\gcp\vm.ps1 <command> [-Gpu]

Commands (in the order you first use them):
    create   Create the VM (no static IP, no open ports)
    upload   Copy the app, model weights, .env and .streamlit/secrets.toml to the VM
    setup    Install everything on the VM and start the app + HTTPS tunnel (prints the link)
    url      Print the current HTTPS link (it changes every time the VM starts)
    stop     Stop the VM (no compute charges while stopped; the disk still costs ~$3/month)
    start    Start the VM again (the app starts by itself; prints the new link)
    logs     Show the app's recent log lines
    ssh      Open a shell on the VM
    delete   Delete the VM and its disk (no further charges)

Two servers, chosen with -Gpu (each has its own disk, so both can exist):
    (default)  child-safety      c3-highcpu-8: 8 vCPU / 16 GB              ~US$0.35-0.40/hour
    -Gpu       child-safety-gpu  n1-standard-4 + NVIDIA T4: 4 vCPU / 15 GB  ~US$0.60/hour
The GPU server needs GPU quota first: Cloud Console > IAM & Admin > Quotas,
"GPUs (all regions)" >= 1. Its first `setup` installs the NVIDIA driver and may
restart the VM once (handled automatically).

After changing code: `upload` then `setup` again (add -Gpu for the GPU server).
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("create", "upload", "setup", "url", "stop", "start", "logs", "ssh", "delete")]
    [string]$Command,
    [switch]$Gpu
)

$Name = if ($Gpu) { "child-safety-gpu" } else { "child-safety" }
$Zone = "asia-southeast1-b"     # Singapore (has T4 GPUs). Pick the zone nearest your users.

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
Write-Host "Server: $Name"

# --strict-host-key-checking=no: skip PuTTY's "store host key?" prompt on Windows
# (the VM's host key changes whenever it is recreated).
$SshOpts = @("--zone", $Zone, "--quiet", "--strict-host-key-checking=no")
function Remote([string]$cmd) { gcloud compute ssh $Name @SshOpts --command $cmd }

# The quick-tunnel link changes on every boot; read the newest one from the tunnel's log.
# Retries while the VM is still booting (SSH refused) or the tunnel has not connected yet.
function Get-AppUrl([int]$tries = 1) {
    $cmd = "journalctl -b -u child-safety-tunnel -o cat --no-pager | grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' | tail -n 1"
    for ($i = 1; $i -le $tries; $i++) {
        $ErrorActionPreference = "Continue"        # ssh errors while booting are expected
        $out = (gcloud compute ssh $Name @SshOpts --command $cmd 2>&1 | Out-String)
        if ($out -match "https://[-a-z0-9]+\.trycloudflare\.com") { return $Matches[0] }
        if ($i -lt $tries) { Write-Host "  waiting for the app to come up ($i/$tries)..."; Start-Sleep -Seconds 10 }
    }
    return $null
}

function Show-Link($u) {
    if ($u) { Write-Host "`nApp link: $u`n(The first page load takes about a minute while the models load.)" }
    else { Write-Warning "The VM is up but there is no link yet. Run 'vm.ps1 url' in a minute, or 'vm.ps1 logs'." }
}

switch ($Command) {
    "create" {
        if ($Gpu) {
            # GPUs cannot live-migrate, so the VM must stop during Google maintenance.
            gcloud compute instances create $Name --zone $Zone `
                --machine-type n1-standard-4 --accelerator "type=nvidia-tesla-t4,count=1" `
                --maintenance-policy TERMINATE `
                --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud `
                --boot-disk-size 40GB --boot-disk-type pd-balanced
        } else {
            gcloud compute instances create $Name --zone $Zone `
                --machine-type c3-highcpu-8 `
                --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud `
                --boot-disk-size 30GB --boot-disk-type pd-balanced
        }
    }
    "upload" {
        $tgz = Join-Path $env:TEMP "child-safety-app.tgz"
        $files = @("app.py", "modules", ".streamlit/config.toml", "requirements.txt", "deploy/gcp/setup_vm.sh",
                   "weights/final_best.pt", "weights/yolov8n.pt", "weights/yolov8s.pt") | Where-Object { Test-Path $_ }
        tar -czf $tgz --exclude "__pycache__" --exclude "modules/tests" @files
        gcloud compute scp $tgz "${Name}:child-safety-app.tgz" @SshOpts
        if (Test-Path ".env") { gcloud compute scp ".env" "${Name}:app.env" @SshOpts }
        else { Write-Warning ".env not found: no Cloudflare TURN on the server." }
        if (Test-Path ".streamlit/secrets.toml") { gcloud compute scp ".streamlit/secrets.toml" "${Name}:secrets.toml" @SshOpts }
        elseif (-not (Select-String -Path ".env" -Pattern "^\s*TELEGRAM_BOT_TOKEN\s*=" -Quiet -ErrorAction SilentlyContinue)) {
            Write-Warning "No Telegram token (TELEGRAM_BOT_TOKEN in .env or .streamlit/secrets.toml): alerts will be off." }
        Remote "mkdir -p ~/app && tar -xzf ~/child-safety-app.tgz -C ~/app && rm ~/child-safety-app.tgz && echo Uploaded."
    }
    "setup" {
        Remote "sed -i 's/\r$//' ~/app/deploy/gcp/setup_vm.sh && bash ~/app/deploy/gcp/setup_vm.sh" | Tee-Object -Variable setupOut
        if (($setupOut | Out-String) -match "REBOOT_NEEDED") {
            Write-Host "Restarting the VM once to load the NVIDIA driver..."
            gcloud compute instances stop $Name --zone $Zone
            gcloud compute instances start $Name --zone $Zone
            Show-Link (Get-AppUrl 24)
            Remote "nvidia-smi -L || echo 'WARNING: GPU still not available, see vm.ps1 logs'"
        }
    }
    "url" {
        $u = Get-AppUrl 1
        if ($u) { Write-Host "App link: $u" } else { Write-Warning "No link yet. Is the VM running? Try again in a minute." }
    }
    "stop" { gcloud compute instances stop $Name --zone $Zone; Write-Host "Stopped. Compute charges have stopped (the disk still costs a little)." }
    "start" {
        gcloud compute instances start $Name --zone $Zone
        Show-Link (Get-AppUrl 18)                   # up to ~3-4 minutes
    }
    "logs" { Remote "journalctl -u child-safety -n 60 -o cat --no-pager" }
    "ssh" { gcloud compute ssh $Name @SshOpts }
    "delete" { gcloud compute instances delete $Name --zone $Zone }
}
