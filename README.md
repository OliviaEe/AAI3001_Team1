# Child Safety Monitor

A team computer-vision prototype combining hazard detection, a person reference, relative-depth/spatial analysis, a Streamlit interface and Telegram caregiver alerts.

## Team and Contributions

**Team:** AAI3001 — Lab P1, Group 1

| Team Member | Admin Number | Contribution |
|:---:|:---:|:---|
| Ee Chew Fong Olivia | 2102743 | Streamlit interface, camera and image/video testing, Telegram alerts and application integration |
| Thet Htar San | 2502046 | Streamlit interface, camera and image/video testing, Telegram alerts and application integration |
| Teng Tze Hui | 2502437 | Dataset preparation, annotation validation, augmentation, model training, experiment evaluation and model-results documentation |
| Fujita Ayumi | 2502515 | Spatial analysis, proximity calculations, risk scoring and spatial integration |
| Low Yi Hang Nicholas | 2500831 | Spatial analysis, proximity calculations, risk scoring and spatial integration |

**Shared contributions:** system integration, testing, project discussions, README, report and slides review.

## Project overview

The prototype addresses camera monitoring of potentially hazardous household objects, including batteries, small items, medicines and sharp tools. It aims to support caregiver awareness while retaining the detection and spatial limitations described below.

This project is a prototype monitoring assistant. It watches a camera feed, image or video and:

1. **Detects household hazards** with a custom-trained YOLOv8 model (seven hazard groups).
2. **Detects people** with a stock COCO person detector, used as the reference point for proximity. For demo purposes, any detected person stands in for the child.
3. **Estimates relative depth** with Depth Anything V2 Small to compare frame-relative depth at the person and hazard boxes. These values are not physical distances.
4. **Scores spatial risk** from the image-plane gap, the depth gap and the hazard type, then displays a `SAFE`, `LOW`, `MEDIUM` or `HIGH` assessment. The spatial explanation separately uses geometric zones named SAFE, CAUTION, WARNING and CRITICAL.
5. **Supports caregiver alerts** through Telegram when the configured level and duration conditions are met, with a cooldown to limit repeated messages.

## Documentation

| Guide | What it covers |
|:---:|:---|
| [UI and application guide](docs/UI_GUIDE.md) | Installation, camera controls, image/video testing, spatial/depth integration, Telegram and deployment |
| [Model experiments, results and limitations](docs/MODEL_RESULTS.md) | Baseline and six experiments, evaluation protocol, selected configuration, findings, limitations, future work and artifact records |
| [Spatial analysis and integration](docs/SPATIAL_ANALYSIS.md) | Relative box distance, depth sampling, risk formula, temporal handling, tests and seven-class risk mapping |
| [Cloud deployment guide](docs/DEPLOYMENT.md) | Streamlit Community Cloud, Google Cloud setup, camera relay, costs and operational checks |
| [Executed notebook](Child_Safety_Hazard_Detection_YOLOv8.ipynb) | Full preparation/training/evaluation code and recorded outputs |

> **Reading the report on GitHub:** the executed notebook is approximately 38 MB and may be too large for GitHub's notebook preview. The [model guide](docs/MODEL_RESULTS.md) is written to stand alone, with readable result tables and figures. Download the notebook and open it in Jupyter or VS Code if you need the full executed cells.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Obtain the selected checkpoint separately and place it at `weights/final_best.pt`, or configure `HAZARD_MODEL_PATH`. Weights are ignored by Git. Without the trained checkpoint, the app uses a stock-model demonstration. If pipeline construction fails, the app displays an error and uses a random-score fallback. Neither is equivalent to the selected seven-group detector. See the [UI guide](docs/UI_GUIDE.md).

## Dataset

### How the dataset was built

The hazard dataset combines **public online images** and **photographs the team took and labelled**:

#### 1. Public datasets (selected sources)

The table below lists some of the public datasets used during data collection. It is not a complete inventory of every source contributing to the final dataset.

| Source | Images/categories sourced | Link |
|:---|:---|:---|
| **ChildSUn** (Kaggle) | Source dataset contains 5,350 images categorised as safe or unsafe for children. This is the source dataset size, not a verified count of images retained in our final dataset. | [kaggle.com/datasets/fairuzkhan002/datasetvgg19](https://www.kaggle.com/datasets/fairuzkhan002/datasetvgg19) |
| **Open Images Dataset V7** | Knives, scissors, kitchen scenes, medicine | [Open Images V7 visualiser](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=all&set=train&c=%2Fm%2F01lsmm) |
| **LVIS** | Knives, scissors, bottles, bottle caps | [platform.ultralytics.com/ultralytics/datasets/lvis](https://platform.ultralytics.com/ultralytics/datasets/lvis) |
| **Baby-Danger** (Roboflow Universe) | Unsafe objects for children | [https://universe.roboflow.com/fyp-ytggz/baby-danger](https://universe.roboflow.com/fyp-ytggz/baby-danger) |
| **Harmful Objects** (Roboflow Universe) | Harmful objects that pose risk | [https://universe.roboflow.com/harmfull-objects/harmful-objects-wmmdi](https://universe.roboflow.com/harmfull-objects/harmful-objects-wmmdi) |

Each source keeps its own licence and terms of use. Check the linked dataset page before redistributing any images. The full image dataset is stored separately. This repository includes split assignments and selected illustrative images, such as the examples below.

#### 2. Self-collected photographs

Team members photographed the hazard objects in real homes: on beds, blankets, clutter, floors and tables, under ordinary indoor lighting. These photos add cluttered backgrounds and everyday phone-camera viewpoints. Their coverage of the intended monitoring conditions still needs evaluation.

<table align="center">
  <tr>
    <td align="center"><img src="figures/dataset/pill_bottle_raw.jpg" alt="Self-collected photo of a pill bottle on a cluttered desk" width="280"><br><em>Raw self-collected photo</em></td>
    <td align="center"><img src="figures/dataset/pill_bottle_labelled.jpg" alt="Same photo with a green bounding box labelled pill_bottle" width="280"><br><em>Labelled Ourselves</em></td>
  </tr>
</table>

#### 3. Annotation

- **Tool:** the team annotated the self-collected photos in tools like **[Roboflow](https://roboflow.com/)**, drawing a bounding box around each hazard object and exporting the labels in YOLO format.
- **Public images:** labels were checked or relabelled where needed so that every source uses the same 14-class scheme.

**Example: from photo to label.** For the pill bottle photo above, the exported YOLO label file ([pill_bottle_label.txt](figures/dataset/pill_bottle_label.txt)) has one line per object:

```text
12 0.552754 0.581933 0.218487 0.231092
```

| Value | Field | Meaning | In pixels (960 × 1280 image) |
|:---:|:---|:---|:---|
| `12` | class ID | `pill_bottle` in the original 14-class scheme | Mapped to final group **6 `medicine`** during preparation |
| `0.552754` | x centre | Box centre ÷ image width | 0.552754 × 960 ≈ **531 px** |
| `0.581933` | y centre | Box centre ÷ image height | 0.581933 × 1280 ≈ **745 px** |
| `0.218487` | width | Box width ÷ image width | 0.218487 × 960 ≈ **210 px** |
| `0.231092` | height | Box height ÷ image height | 0.231092 × 1280 ≈ **296 px** |

All four coordinates are **normalised to 0–1**, so the label stays valid if the image is resized. The box corners are `(x1, y1) ≈ (426, 597)` and `(x2, y2) ≈ (636, 893)`. 

### Classes

The original annotations use **14 classes**, which are grouped into **7 hazard groups** for the final detector:

| Final ID | Hazard group | Original classes |
|:---:|:---|:---|
| 0 | `battery` | battery |
| 1 | `coin_battery` | coin_battery |
| 2 | `coin` | coin |
| 3 | `small_object` | bottlecap, dice, marble, small_toy |
| 4 | `pen` | pen, pencap |
| 5 | `sharp` | knife, scissors |
| 6 | `medicine` | medicine_syrup, pill_bottle, pill_packet |

Grouping reflects the intended hazard categories. It is a design choice, not a demonstrated accuracy improvement. Predictions identify the group rather than the original item, such as knife versus scissors. The current spatial mapping recognises all seven groups; their risk weights, including the combined pen/pen-cap group, still require validation.

### Cleaning and splitting

| Step | Result |
|:---|:---|
| Raw images | 18,073 |
| Removed exact duplicates | 923 |
| Removed duplicates with conflicting annotations | 14 |
| Removed empty-label images | 2 |
| **Prepared images** | **17,134** |

- **Leakage control:** near-duplicate photos are grouped by visual hashing, and each photo group stays in a single split.
- **Split:** 11,996 train / 2,570 validation / 2,568 test images (20,184 / 4,370 / 4,173 objects).
- **Training-only additions:** underrepresented classes are oversampled, background (negative) images and object-paste examples are added, and online augmentation varies scale, framing, lighting, blur and noise. Validation and test images are not augmented. Separate *stress* copies of validation and test apply synthetic degradation for robustness evaluation.

Per-image split assignments are in [results/split_membership.csv](results/split_membership.csv). The images themselves are not in Git. See [data preparation details](docs/MODEL_RESULTS.md#why-data-preparation-matters).

## Selected hazard detector

**Experiment 3: YOLOv8s, input size 640, confidence threshold 0.30, detector-only.** It has the highest mAP@50–95 among the six detectors across clean validation, validation stress, clean test and test stress. Clean-test mAP@50–95 is **0.6549**, and test-stress mAP@50–95 is **0.6080**.

Experiment 6 obtains **96.63% crop-validation accuracy**, but adding it to the detector increases missed hazards and false detections. It is retained as an experiment, not recommended for the final operating configuration.

## Experiment records in this repository

- `results/validation/`: recorded validation scores and threshold sweeps.
- `results/final_test/`: detector, pipeline, class, size and controlled-condition CSV results.
- `results/training_history/`: original epoch histories and run arguments for the completed baseline and Experiments 1–6.
- `results/split_membership.csv`: exported filenames, categories, photo groups and split assignments; no image files.
- `figures/model_training/`: selected evaluation and training figures.
- `configs/model_training/selected_detector.json`: recommended settings and seven-class mapping. This is a record, not an automatically loaded app configuration.
- `configs/model_training/artifact_manifest.json`: source runs, checkpoint hashes, notebook hash and copied-record checksums.

Large datasets, generated image folders and model checkpoints remain outside normal Git tracking. Their locations and access requirements are documented in the [model guide](docs/MODEL_RESULTS.md#artifact-locations-and-verification).

## System scope and integration status

```text
Camera/image/video -> trained hazard detector ----+
                  -> COCO person detector --------+-> relative spatial assessment -> UI / optional alerts
                  -> relative-depth estimation --+
```

The selected checkpoint supplies seven hazard groups. The current spatial mapping includes all seven, including `small_object`, `sharp` and `medicine`. Group coverage is implemented, while the risk policy still requires validation. In particular, the combined `pen` group includes pen caps but retains the legacy pen weight. See [current label handling](docs/SPATIAL_ANALYSIS.md#seven-class-integration-and-remaining-checks).

The model study evaluates hazard detection on annotated images. The [spatial guide](docs/SPATIAL_ANALYSIS.md) explains proximity assumptions, depth freshness and score persistence; the [UI guide](docs/UI_GUIDE.md) covers camera operation and alert configuration.

## Known Limitations

### Detection trade-off

The project prioritises recovering hazards while monitoring false detections. The confidence threshold `0.30` was selected on validation to minimise false detections per image among tested thresholds meeting overall recall of at least `0.85`. This is a project criterion, not a validated safety standard. False alerts can also distract or fatigue caregivers, so their cost should not be dismissed.

### Hazard detection

| Selected detector at confidence 0.30 | Clean test | Synthetic stress |
|:---|:---:|:---:|
| False detections | 546 | 366 |
| Missed objects | 469 | 721 |

- **Errors in both directions remain.** The clean test contains 2,568 images. These are object-level counts, not caregiver-alert counts. Under stress, fewer false detections occur alongside more missed hazards.
- **Small objects remain difficult.** Recall is about `0.65` for objects with a shorter side of 16–32 model-input pixels, compared with `0.92` at 64+ pixels. These measurements do not establish a physical operating distance.
- **Some classes are weaker.** Stress recall is approximately `0.70` for pen, `0.75` for sharp objects and `0.77` for battery.
- **Camera conditions can differ from the test data.** Synthetic blur, noise and lighting changes do not replace independent testing in representative homes. A frame with no detections does not establish that the scene is safe.

See [model failures and limitations](docs/MODEL_RESULTS.md#failure-analysis-and-practical-consequences) for class, size and confidence analyses.

### Spatial analysis and relative depth

- **Proximity is relative, not measured in metres.** The image-plane calculation divides box-edge separation by the person's box height. Posture, camera angle and truncated detections affect it; overlapping boxes do not establish contact.
- **Depth is relative, not metric.** The pipeline normalises the depth map within each frame and samples the central part of each detection box. The person's sampled region does not necessarily represent a reaching hand. Reflective surfaces, transparent objects and very small objects require additional evaluation.
- **Depth can describe an earlier frame.** Live assessment accepts a depth map only while its source-frame age is at most `1.5 seconds`; older maps fall back to image-plane proximity. Even an accepted map can lag a moving scene.
- **Displayed scores can persist.** A one-second score hold and smoothing reduce flicker but delay clearance after the raw risk falls. This is separate from model processing latency.
- **Risk weights are manually assigned.** Hazard weights, distance thresholds and the `0.30` foreground-held-object depth allowance have not been validated against physical reach or injury risk. All seven labels are recognised, but the merged pen/pen-cap group still uses the legacy pen weight of `0.30` and needs review.
- **A detected person is only a reference.** The model does not identify age. SAFE is a computed application state, not a safety certification.

See [spatial assumptions and remaining checks](docs/SPATIAL_ANALYSIS.md#seven-class-integration-and-remaining-checks) for formulas, timing behaviour and implementation details. Detector evaluation alone does not validate the complete camera/depth/alert pipeline.

## Cloud Deployment

**Streamlit Community Cloud is the main deployment. Google Cloud is the backup**, for when Streamlit Community Cloud is unavailable or too slow. Both use the same **Cloudflare TURN** relay for the live camera.

| | Streamlit Community Cloud (main) | Google Cloud (backup) |
|:---|:---|:---|
| Cost | Free | Pay per hour while running (stop it after use) |
| Hardware | Shared, small | Dedicated 8 vCPU server, or an NVIDIA T4 GPU server |
| Link | Fixed `*.streamlit.app` address | New `*.trycloudflare.com` address on every start |
| Setup | Deploys from GitHub | Scripts in [`deploy/gcp/`](deploy/gcp/) |
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
[cloud camera setup](docs/UI_GUIDE.md#cloud-camera-setup).

Cloud testing confirmed that uploaded-image inference, live camera monitoring, and Telegram alert delivery work. A teammate who previously could not connect their camera successfully tested live monitoring after Cloudflare TURN was configured. Processing speed depends on the available cloud resources and network connection. TURN improves connectivity but does not increase the app's CPU resources. Local database storage is not guaranteed to persist across cloud rebuilds or restarts.

### Google Cloud (backup)

If Streamlit Community Cloud is down, asleep or too slow for a demo, the same `app.py` can run on a Google Cloud Compute Engine virtual machine (VM). It provides dedicated resources and incurs charges while running. The performance table below records the team's CPU-server measurements; results depend on workload and hardware. One PowerShell script on your PC, [`deploy/gcp/vm.ps1`](deploy/gcp/vm.ps1), creates, updates, starts and stops it. The VM opens **no inbound ports**:

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
3. **`.env` file.** Copy [`.example.env`](.example.env) to `.env` and fill in the values below. `.env` is ignored by Git. Never commit keys or tokens.
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

#### Measured processing latency (CPU server)

Team-recorded timings from the live app (**Performance details → Latency**). These measurements describe the processing shown by that panel, not camera-to-notification latency or a guaranteed frame rate:

| `MODEL_THREADS` | Reported processing latency per frame |
|:---:|:---:|
| 2 (laptop default) | 140–190 ms |
| **4 (VM default)** | **100–130 ms** |
| 6 | 115–135 ms |

In these measurements, using more threads did not consistently reduce latency. Detection and background depth processing compete for CPU resources. The **Max model runs / second** setting controls inference scheduling; increasing it does not guarantee an equivalent camera frame rate. Capture, network transport, rendering and alert delivery add separate delays.

#### Notes

- **One Telegram poller per bot token.** Do not run the Google Cloud VM, Streamlit Community Cloud and a local copy with the same `TELEGRAM_BOT_TOKEN` at the same time, or bot commands will conflict.
- **Subscriptions persist** on the VM's disk (`subscribers.db`) across stop and start, unlike Streamlit Community Cloud. They are lost if the VM is deleted.
- **TURN credentials reach the browser.** They are short-lived (at most about 2 hours). Deleting the TURN key in Cloudflare disables the relay immediately.

For a separate setup reference, see the [deployment guide](docs/DEPLOYMENT.md).
