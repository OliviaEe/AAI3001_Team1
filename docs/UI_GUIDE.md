# UI and Application Guide

Streamlit UI for live camera monitoring, image/video testing, and Telegram alerts.

[Project overview](../README.md) · [Model results and limitations](MODEL_RESULTS.md) · [Spatial analysis](SPATIAL_ANALYSIS.md) · [Deployment](DEPLOYMENT.md)

**Integration status:** the trained model outputs seven hazard groups, and the current spatial mapping recognises all seven alongside legacy labels. Their manually assigned risk weights still need validation, especially the combined pen/pen-cap group. The Experiment 6 crop classifier is excluded from the recommended hazard-detection configuration.

Trained pipeline (`HazardSpatialClassifier`), used when the selected checkpoint is available through a configured or default path:

```text
camera frame -> trained YOLOv8 (7 hazard groups)         -> hazards  -+
             -> stock YOLOv8n (COCO "person")             -> person   -+-> SpatialAnalyzer
             -> Depth Anything V2 (background thread)     -> depth    -+   -> danger level
```

Put the final evaluation's selected model, `final_best.pt`, in `weights/` and restart. See [artifact locations and verification](MODEL_RESULTS.md#artifact-locations-and-verification). `HAZARD_MODEL_PATH` can point elsewhere. Without it, the
app uses the stock-YOLO demonstration (`SpatialDemoClassifier`: person, knife and scissors only). The Live page identifies the active pipeline. If pipeline construction fails, it shows the error and uses `DummyClassifier`, which produces random scores.

`weights/yolov8n.pt` is the person model used in the trained-hazard path; the stock-only demo uses `weights/yolov8s.pt`. Missing stock weights download on first use, as does the depth model (requires `transformers` and internet for initial setup). A person stands in for the child reference; this model does not identify age.

The UI owns one browser camera stream; it imports the demo's helper functions
without running its camera-opening `main()`. Live depth runs in the background
with one outstanding frame, and a one-second `ScoreHold` is per stream.
Image/video tests compute depth from each tested frame. There is no object tracker in this pipeline.

CPU depth can take several seconds. The proximity panel shows the map's age;
video and detection can stay live while depth-based distance updates lag behind.
Earlier UI notes recorded map ages of approximately 5–15 seconds. In the current implementation, maps older than **1.5 seconds** are excluded from proximity scoring and the app falls back to image-plane distance. The old map can remain visible with its age; this does not mean it is used for assessment. A usable map can still describe an earlier frame within the freshness window. See the [live timing explanation](SPATIAL_ANALYSIS.md#live-timing-fallback-and-score-persistence). Duplicate detections and live proximity
accuracy still need model/pipeline validation; functional UI integration does
not establish that a displayed SAFE level describes the current scene accurately.
Distinguish this timing limitation from the spatial risk formula when refining
the pipeline. Distances are relative box-height units, not metres.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The interface in `app.py` provides live camera controls, image and video testing, and four
separate status cards for Assessment, Detected objects, Proximity assessment,
and Caregiver alerts. The assessment displays the spatial score and explanation.
The Settings button is on the navigation row. With `DummyClassifier`, enabled automatic alerts use random scores and serve only as a demonstration. Manual test messages are also available.

The combined training report records separate Kaggle runs. Its saved outputs can be read without rerunning training. Reproduction requires the raw dataset and evaluation bundle; see [model records](MODEL_RESULTS.md#reproduction-and-records). The root `requirements.txt` contains app and training dependencies; the recorded Kaggle environment differs from the local app environment.

## Integration

```text
BGR image → detector.detect() → spatial.assess() → ClassificationResult → UI/alerts
```

### Hazard Detection

`YOLODetector` in `modules/detector.py` handles inference with the trained hazard model. The implementation loads the weights once in `__init__()` and reuses them.

`detect(frame_bgr)` accepts an OpenCV BGR image and returns a `list[Detection]`. Each detection contains:

- A class label.
- A confidence score in `[0, 1]`.
- A bounding box `(x1, y1, x2, y2)` in the original image’s pixel coordinates.

Keep data augmentation in training, rather than applying random augmentation during inference.

### Spatial Assessment

`SpatialAnalyzer.assess(detections, frame_shape)` in `modules/spatial.py` receives detections and the frame dimensions `(height, width)`. Its interface returns `(danger_score, reason)`, where the score is in `[0, 1]` and the reason explains the assessed risk.

The analyzer excludes detections without boxes. With no reference person or no recognised hazard, its raw score is zero and its reason states the missing input. That score does not establish physical safety; see the spatial guide for assumptions about box geometry. Image-based proximity does not directly measure physical distance. A person detection provides a reference person; it does not establish that the person is a child.

Detection confidence and danger score are separate values.

### Pipeline and Label Mapping

`SafetyClassifier` connects the detector and spatial interfaces and converts the score into `SAFE`, `LOW`, `MEDIUM` or `HIGH`. The UI calls:

```python
predict(frame_bgr) -> ClassificationResult
```

The result contains `level`, `score`, `detections` and `reason`.

`get_classifier()` in `modules/classifier.py` selects the pipeline:

- `HazardSpatialClassifier`: trained hazards, person detection, depth and spatial assessment when trained weights are available.
- `SpatialDemoClassifier`: stock-model demonstration when trained weights are unavailable.
- `DummyClassifier`: random-score fallback selected by the app when pipeline construction fails.

The final hazard model outputs seven groups: `battery`, `coin_battery`, `coin`, `small_object`, `pen`, `sharp` and `medicine`.

The spatial mapping includes all seven groups as well as older labels. Its explicit entries cover `small_object`, `sharp` and `medicine`. The combined `pen` group retains weight `0.30`, so pen caps receive that same treatment. See [implemented weights and remaining checks](SPATIAL_ANALYSIS.md#seven-class-integration-and-remaining-checks). The legacy `medicine_syrup` alias is separate from the final `medicine` entry.

### Integration Checks

Use **Test Image** and **Test Video** to inspect bounding boxes, labels, confidence scores and risk explanations.

Run the pipeline tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest modules/tests/test_depth_logic.py modules/tests/test_hazard_pipeline.py -q
```

Passing functional tests does not establish reliable real-camera hazard detection or child identification.

## Telegram alerts

The project bot is **CVDL Child Safety** ([@childsafetyalertcvdl_bot](https://t.me/childsafetyalertcvdl_bot)).
Create `.streamlit/secrets.toml` (already ignored by Git):

```toml
TELEGRAM_BOT_TOKEN = "your_bot_token"
# Optional fallback for invite links if the getMe lookup fails
TELEGRAM_BOT_USERNAME = "childsafetyalertcvdl_bot"
```

How recipients are organised:

```text
Home (one household, private home key)
 ├─ Rooms:   Living room, Bedroom, ...   <- each camera picks its room in Settings
 └─ Members: Telegram chats that joined with the home's invite link
             (a parent, or a family group chat)
```

1. Open the **Telegram** tab and create a home. Its private key is put in the
   page URL, so bookmark the page (or keep the key) to reopen it.
2. Add one room per camera location.
3. Send the invite link to caregivers. One tap joins **all** rooms of the home.
4. On **Live Monitoring**, under Caregiver alerts, pick this camera's room and
   switch on **Send Telegram alerts**.
5. Use **Send test message** / **Send test alert** in the Telegram tab to check delivery.

In Telegram, members can use `/rooms` to mute or unmute single rooms, or leave a
home, and `/stop` to leave everything. One person can join several homes
(e.g. their own and a grandparent's) and gets alerts labelled `Home · Room`.

Home and subscription handling:

- the home key selects the displayed rooms and subscriptions; keep it private because anyone with the key can access that home;
- SQLite (`subscribers.db`) stores subscriptions; large-scale or multi-server behaviour requires separate testing;
- alerts go through one queue that stays under Telegram's rate limit and retries
  when Telegram asks to slow down; a photo is uploaded once and re-used for all members;
- chats that blocked the bot are removed automatically;
- the alert cooldown is per room and shared, so two open browsers don't double-alert.

With `DummyClassifier`, alerts can be switched on for testing but use random
scores (set **Level must last** to 0 in Settings to trigger one quickly).
`.streamlit/config.toml` forces the light theme so text stays readable. Do not commit bot secrets.

## Deployment

See the [deployment guide](DEPLOYMENT.md) for the live application, Streamlit Community Cloud setup and Google Cloud backup commands. The camera-relay settings below apply to the hosted interface.

WebRTC uses the viewer's browser camera. For Streamlit Community Cloud, point
the deployment to `app.py` and enter the bot token in App settings -> Secrets.
Model hosting requirements depend on the selected model. For remote camera
connection issues, configure Cloudflare TURN as described below.

Subscriptions are stored in local `subscribers.db` (SQLite). Hosts that wipe the disk
on restart (e.g. Streamlit Community Cloud) need persistent storage or a hosted
database (Postgres/Supabase) behind the same `SubscriptionStore` methods. Only one
running server may poll the bot; with several servers, switch the bot to a webhook.

### Cloud camera setup

Create a **TURN Server app** in Cloudflare, then add its credentials to
**Streamlit App settings > Secrets**, keeping the existing Telegram setting:

```toml
CLOUDFLARE_TURN_KEY_ID = "your-turn-token-id"
CLOUDFLARE_TURN_KEY_API_TOKEN = "your-api-token"
```

Remove any `METERED_TURN_*` settings when switching providers: they take precedence
over Cloudflare. Save and reboot the app, then check **Performance details** for
**Cloudflare TURN available** and test the camera on the affected device. This
status confirms relay credentials were obtained; video may still connect directly.
Keep credentials private and stop monitoring after testing. TURN improves
connectivity; it does not increase Streamlit's CPU resources.

## Files and ownership

<table align="center">
  <thead>
    <tr><th align="center">File</th><th align="center">Purpose</th><th align="center">Owners</th></tr>
  </thead>
  <tbody>
    <tr><td align="center"><code>app.py</code></td><td align="center">Improved UI, camera controls, four status cards, image &amp; video tester, Telegram setup, Settings, and alert history</td><td align="center">Liv &amp; Ren</td></tr>
    <tr><td align="center"><code>modules/camera.py</code></td><td align="center">Browser camera frames, background inference, overlays, and video analysis</td><td align="center">Liv &amp; Ren</td></tr>
    <tr><td align="center"><code>modules/notifier.py</code></td><td align="center">Telegram send queue (rate limits, photo reuse) and per-room cooldown</td><td align="center">Liv &amp; Ren</td></tr>
    <tr><td align="center"><code>modules/subscriptions.py</code></td><td align="center">Homes, rooms, members (SQLite) and bot commands</td><td align="center">Liv &amp; Ren</td></tr>
    <tr><td align="center"><code>modules/detector.py</code></td><td align="center"><code>YOLODetector</code> for the trained hazard model; finds <code>final_best.pt</code></td><td align="center">Tze</td></tr>
    <tr><td align="center"><code>modules/spatial.py</code></td><td align="center">Spatial interface and proximity/risk calculation</td><td align="center">Ayumi &amp; Nic</td></tr>
    <tr><td align="center"><code>modules/classifier.py</code></td><td align="center">Shared result types and pipeline integration</td><td align="center">Shared</td></tr>
    <tr><td align="center"><code>Child_Safety_Hazard_Detection_YOLOv8.ipynb</code></td><td align="center">Augmentation, training, and evaluation</td><td align="center">Tze</td></tr>
  </tbody>
</table>
