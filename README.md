# Child Safety Monitor

A team computer-vision prototype combining hazard detection, a person reference, relative-depth/spatial analysis, a Streamlit interface and Telegram caregiver alerts.

## Team and Contributions

**Team:** AAI3001 — Lab P1, Group 1

| Team Member | Admin Number | Contribution |
|:---:|:---:|:---:|
| Ee Chew Fong Olivia | 2102743 | Streamlit interface, camera and image/video testing, Telegram alerts and application integration |
| Thet Htar San | 2502046 | Streamlit interface, camera and image/video testing, Telegram alerts and application integration |
| Teng Tze Hui | 2502437 | Dataset preparation, annotation validation, augmentation, model training, experiment evaluation and model-results documentation |
| Fujita Ayumi | 2502515 | Spatial analysis, proximity calculations, risk scoring and spatial integration |
| Low Yi Hang Nicholas | 2500831 | Spatial analysis, proximity calculations, risk scoring and spatial integration |

**Shared contributions:** system integration, testing, project discussions and report review.

## Documentation

| Guide | What it covers |
|:---:|:---:|
| [UI and application guide](docs/UI_GUIDE.md) | Installation, camera controls, image/video testing, spatial/depth integration, Telegram and deployment |
| [Model experiments, results and limitations](docs/MODEL_RESULTS.md) | Baseline and six experiments, evaluation protocol, selected configuration, findings, limitations, future work and artifact records |
| [Spatial analysis and integration](docs/SPATIAL_ANALYSIS.md) | Relative box distance, depth sampling, risk formula, temporal handling, tests and seven-class integration gap |
| [Executed notebook](Child_Safety_Hazard_Detection_YOLOv8.ipynb) | Full preparation/training/evaluation code and recorded outputs |

> **Reading the report on GitHub:** the executed notebook is approximately 38 MB and may be too large for GitHub's notebook preview. The [model guide](docs/MODEL_RESULTS.md) is written to stand alone, with readable result tables and figures. Download the notebook and open it in Jupyter or VS Code if you need the full executed cells.

## Selected hazard detector

**Experiment 3: YOLOv8s, input size 640, confidence threshold 0.30, detector-only.** It has the highest mAP@50–95 among the six detectors across clean validation, validation stress, clean test and test stress. Clean-test mAP@50–95 is **0.6549**, and test-stress mAP@50–95 is **0.6080**.

Experiment 6 obtains **96.63% crop-validation accuracy**, but adding it to the detector increases missed hazards and false detections. It is retained as an experiment, not recommended for the final operating configuration.

## System scope and integration status

```text
Camera/image/video -> trained hazard detector ----+
                  -> COCO person detector --------+-> relative spatial assessment -> UI / optional alerts
                  -> relative-depth estimation --+
```

The person detector does not establish age. Relative proximity is not a distance measurement in metres, and absence of detections is not proof of safety.

**Integration work remains:** the selected model outputs seven hazard groups, while the current app spatial weights still use the legacy fourteen-class labels. The spatial filter currently excludes `small_object`, `sharp` and `medicine`; the merged `pen` group retains the old pen weight. See the [spatial integration gap](docs/SPATIAL_ANALYSIS.md#seven-class-integration-gap) before relying on risk levels. Model evaluation does not validate the complete live camera/depth/alert pipeline. App code is unchanged by this documentation update.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Obtain the selected checkpoint separately and place it at `weights/final_best.pt`, or configure `HAZARD_MODEL_PATH`. Weights are ignored by Git. Without the trained checkpoint, the app uses its stock-model demo/fallback as described in the [UI guide](docs/UI_GUIDE.md).

## Experiment records in this repository

- `results/validation/`: recorded validation scores and threshold sweeps.
- `results/final_test/`: detector, pipeline, class, size and controlled-condition CSV results.
- `results/training_history/`: original epoch histories and run arguments for the completed baseline and Experiments 1–6.
- `results/split_membership.csv`: exported filenames, categories, photo groups and split assignments; no image files.
- `figures/model_training/`: selected evaluation and training figures.
- `configs/model_training/selected_detector.json`: recommended settings and seven-class mapping. This is a record, not an automatically loaded app configuration.
- `configs/model_training/artifact_manifest.json`: source runs, checkpoint hashes, notebook hash and copied-record checksums.

Large datasets, generated image folders and model checkpoints remain outside normal Git tracking. Their locations and access requirements are documented in the [model guide](docs/MODEL_RESULTS.md#artifact-locations-and-verification).
