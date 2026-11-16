# UI and Application Guide

Streamlit UI for live camera monitoring, image/video testing, and Telegram alerts.

[Project overview](../README.md) · [Model results and limitations](MODEL_RESULTS.md) · [Spatial analysis](SPATIAL_ANALYSIS.md)

**Integration status:** the final trained model has seven hazard groups. The existing spatial risk mapping still contains the older fourteen-class names; integration of the grouped labels needs review before relying on its risk levels. The trained crop classifier from Experiment 6 is not part of the recommended detection configuration.

Full pipeline (`HazardSpatialClassifier`), used when `weights/final_best.pt` exists:

```text
camera frame -> Tze's trained YOLOv8 (7 hazard groups)  -> hazards  -+
             -> stock YOLOv8n (COCO "person")             -> person   -+-> SpatialAnalyzer
             -> Depth Anything V2 (background thread)     -> depth    -+   -> danger level
```

Put the final evaluation's selected model, `final_best.pt`, in `weights/` and restart. See [artifact locations and verification](MODEL_RESULTS.md#artifact-locations-and-verification). `weights/` is not in Git, so each
person copies it once; `HAZARD_MODEL_PATH` can point elsewhere. Without it, the
app uses the spatial team's stock-YOLO demo (`SpatialDemoClassifier`: person,
knife and scissors only). The Live page shows which one is running, and a
model that fails to load is shown as an error with demo mode.

`weights/yolov8n.pt` is the person model used in the trained-hazard path; the stock-only demo uses `weights/yolov8s.pt`. Missing stock weights download on first use, as does the depth model (requires `transformers` and internet for initial setup). A person stands
in for the child reference; this model does not identify age.

The UI owns one browser camera stream; it imports the demo's helper functions
without running its camera-opening `main()`. Live depth runs in the background
with one outstanding frame, and the team's one-second `ScoreHold` is per stream.
Image/video tests compute depth from each tested frame. The spatial and depth
source files remain unchanged. No object tracker or additional Python module
was added.

CPU depth can take several seconds. The proximity panel shows the map's age;
video and detection can stay live while depth-based distance updates lag behind.
Earlier UI notes recorded map ages of approximately 5–15 seconds. In the current implementation, maps older than **1.5 seconds** are excluded from proximity scoring and the app falls back to image-plane distance. The old map can remain visible with its age; this does not mean it is used for assessment. A usable map can still describe an earlier frame within the freshness window. See the [live timing explanation](SPATIAL_ANALYSIS.md#live-timing-fallback-and-score-persistence). Duplicate detections and live proximity
accuracy still need model/pipeline validation; functional UI integration does
not establish that a displayed SAFE level describes the current scene accurately.
Distinguish this timing limitation from the spatial risk formula when refining
the pipeline. Distances are relative box-height units, not metres.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The improved UI is in `app.py`: live camera controls, image and video testing, and four
separate status cards for Assessment, Detected objects, Proximity assessment,
and Caregiver alerts. The assessment displays the team's score and explanation.
The Settings button is on the navigation row. Automatic Telegram alerts are
disabled while `DummyClassifier` is selected; manual test messages are available.

The combined training report records separate Kaggle runs. Its saved outputs can be read without rerunning training. Reproduction requires the raw dataset and evaluation bundle; see [model records](MODEL_RESULTS.md#reproduction-and-records). The root `requirements.txt` contains app and training dependencies; the recorded Kaggle environment differs from the local app environment.

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

## Integration

```text
BGR image -> detector.detect() -> spatial.assess() -> ClassificationResult -> UI/alerts
```

**Tze:** implement `YOLODetector` in `modules/detector.py`. Load weights once in
`__init__()`. `detect(frame_bgr)` accepts an OpenCV BGR image and returns a
`list[Detection]`, with labels, confidence in `[0, 1]`, and bounding boxes
`(x1, y1, x2, y2)` in the original image's pixels. Keep augmentation in training.

**Ayumi & Nic:** implement `SpatialAnalyzer.assess(detections, frame_shape)` in
`modules/spatial.py`. `frame_shape` is `(height, width)`. Return
`(danger_score, reason)`, where the score is in `[0, 1]` and the reason explains
the risk. Define proximity calculations and behaviour when children, hazards,
or bounding boxes are missing. Image distance alone is not physical distance.

Agree on class labels and how children are identified before integration.
A person detection alone does not identify a child. Detection confidence and
danger score are separate values.

**Shared:** `SafetyClassifier` already connects these interfaces and converts
the score to `SAFE`, `LOW`, `MEDIUM`, or `HIGH`. The UI continues to call
`predict(frame_bgr) -> ClassificationResult`, containing `level`, `score`,
`detections`, and `reason`.

`get_classifier()` in `modules/classifier.py` picks the pipeline:
`HazardSpatialClassifier` (trained hazards + person + depth + spatial) when the
trained weights are found, otherwise `SpatialDemoClassifier`. `DummyClassifier`
remains the labelled fallback if model loading fails. The existing adapter renames legacy `medicine_syrup` to `liquid_medicine`. The final model instead outputs grouped labels including `medicine`, `sharp` and `small_object`; those require explicit spatial-weight handling. The combined `pen` group also includes pen caps, so the older pen-only risk weight needs review. Use the **Test Image** / **Test Video** tabs to check boxes,
scores, and explanations. Tests:
`python -m pytest modules/tests/test_depth_logic.py modules/tests/test_hazard_pipeline.py -q`.

## Telegram alerts

Create `.streamlit/secrets.toml` (already ignored by Git):

```toml
TELEGRAM_BOT_TOKEN = "your-bot-token"
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

Built for many users:
- every browser only sees its own home; nobody can see or delete other homes;
- subscriptions are in SQLite (`subscribers.db`), safe with many users and threads;
- alerts go through one queue that stays under Telegram's rate limit and retries
  when Telegram asks to slow down; a photo is uploaded once and re-used for all members;
- chats that blocked the bot are removed automatically;
- the alert cooldown is per room and shared, so two open browsers don't double-alert.

With `DummyClassifier`, alerts can be switched on for testing but use random
scores (set **Level must last** to 0 in Settings to trigger one quickly).
`.streamlit/config.toml` forces the light theme so text stays readable. Do not commit bot secrets.

## Deployment

WebRTC uses the viewer's browser camera. For Streamlit Community Cloud, point
the deployment to `app.py` and enter the bot token in App settings -> Secrets.
Model hosting requirements depend on the selected model. Restrictive networks
may require a TURN server in `RTC_CONFIG` in `app.py`.

Subscriptions are stored in local `subscribers.db` (SQLite). Hosts that wipe the disk
on restart (e.g. Streamlit Community Cloud) need persistent storage or a hosted
database (Postgres/Supabase) behind the same `SubscriptionStore` methods. Only one
running server may poll the bot; with several servers, switch the bot to a webhook.
