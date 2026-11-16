# Model Experiments, Results and Limitations

[Project overview](../README.md) · [UI guide](UI_GUIDE.md) · [Executed notebook](../Child_Safety_Hazard_Detection_YOLOv8.ipynb)

This document summarises the final notebook, particularly Sections 6–10, and links its supporting records. It reports completed measurements, not an additional training run. The notebook's code and saved results are unchanged by this documentation update.

## Experiment design and selection

The source annotations have fourteen classes, mapped to seven hazard groups: `battery`, `coin_battery`, `coin`, `small_object`, `pen`, `sharp`, and `medicine`. The prepared split contains **11,996 training, 2,570 validation and 2,568 test images**, with **20,184 / 4,370 / 4,173 objects** respectively. Photo-group assignments stay within one split. Runs were executed separately, with matching exported split membership.

All six detectors use the same prepared data, balancing-plus-extras recipe and augmentation strengths. Training uses AdamW, a cosine schedule, seed 42, batch 8, maximum 40 epochs and patience 15. The baseline learning rate is 0.0005. The balanced training recipe contains 12,352 original-image entries (including repetitions), 600 background images and 1,500 object-paste images. Repetitions are not new independent images.

| Run | Main change | Input | Primary reference |
|---|---|---:|---|
| Baseline | YOLOv8n, learning rate 0.0005 | 640 | Reference |
| Exp1 | Learning rate 0.0001 | 640 | Baseline |
| Exp2 | YOLOv8n at higher resolution | 960 | Baseline |
| Exp3 | Larger YOLOv8s model | 640 | Baseline |
| Exp4 | YOLOv8s-P2 finer prediction grid | 640 | Exp3 |
| Exp5 | Higher-resolution YOLOv8s-P2 | 960 | Exp4 |
| Exp6 | YOLOv8n crop classifier, learning rate 0.001, 25 epochs | 96-pixel crops | Selected detector alone versus detector + classifier |

P2 uses compatible pretrained-weight transfer and has additional layers requiring training. Comparisons assess the implemented configurations under the shared budget, not architecture in isolation.

**Selection protocol:** choose the detector by clean-validation mAP@50–95. On validation, choose the confidence threshold with fewest false boxes per image among tested thresholds meeting overall recall >= 0.85. Exp3 with confidence 0.30 meets that validation target. Classifier rejection thresholds are also evaluated on validation; none meets the target, so 0.20 is provisional. Test data assess the fixed configurations, not a new threshold search.

## Detector results

All numbers below are **mAP@50–95**, higher is better.

| Model | Validation | Validation stress | Test | Test stress |
|---|---:|---:|---:|---:|
| Baseline | 0.6032 | 0.5585 | 0.6052 | 0.5662 |
| Exp1 | 0.5836 | 0.5526 | 0.5996 | 0.5565 |
| Exp2 | 0.4909 | 0.5364 | 0.4893 | 0.5437 |
| **Exp3** | **0.6441** | **0.5945** | **0.6549** | **0.6080** |
| Exp4 | 0.6279 | 0.5880 | 0.6382 | 0.5929 |
| Exp5 | 0.5248 | 0.5405 | 0.5193 | 0.5518 |

Source: [final model comparison](../results/model_training/final_test/final_model_comparison.csv). Increasing model capacity is the only tested change improving this metric against its intended reference across all four sets. Exp3's clean-test gain over baseline is **4.97 percentage points**. P2 at 640 is second-best overall; higher resolution and lower learning rate do not improve aggregate mAP under this recipe.

mAP is supplied by YOLO's evaluator. The following operating-point results use separate class-aware, one-to-one matching at IoU 0.50 and detector confidence 0.30. These values need not equal YOLO's automatically reported precision/recall during training validation.

## Why Experiment 6 is not retained

The classifier's **96.63% accuracy** measures labelling of supplied validation crops. It is not full-image detection mAP and cannot be directly compared with Exp3's mAP. A classifier cannot recover hazards absent from detector proposals.

| Evaluation | Method | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| Clean test | Exp3 alone | 0.872 | 0.888 | 0.879 | 546 | 469 |
| Clean test | Exp3 + classifier | 0.863 | 0.861 | 0.862 | 571 | 578 |
| Stress | Exp3 alone | 0.904 | 0.827 | 0.864 | 366 | 721 |
| Stress | Exp3 + classifier | 0.791 | 0.595 | 0.679 | 656 | 1691 |

Source: [pipeline comparison](../results/model_training/final_test/detector_pipeline_comparison.csv). Adding the classifier introduces **109 additional missed objects and 25 additional false detections on clean test**, and **970 additional misses and 290 additional false detections under stress**. It reclassifies proposals as well as rejecting them; changing a correct class to an incorrect one can create both an FP and an FN. Detector-only operation is supported under these measured settings. Crop/proposal distribution differences are plausible contributors but are not proven causes. Other detector–classifier pairings were not exhaustively tested.

## Findings and limitations

| Finding / limitation | Evidence and scope |
|---|---|
| Small objects remain difficult | Exp3 clean-test recall is 0.649 at 16–32 input pixels versus 0.917 at 64+ pixels (shorter box side). The under-8 and 8–16 clean-test groups have only 3 and 6 objects, so conclusions there are uncertain. |
| Degraded conditions reduce recovery | Exp3 stress recall is 0.827 versus 0.888 on clean test; meeting the validation target does not guarantee stress recall >= 0.85. |
| Uneven hazard-class recall | Pen, sharp and battery stress recall are 0.702, 0.748 and 0.767. An aggregate target does not guarantee every class meets it. |
| Severe shrinkage is a major weakness | On the controlled 150-image subset, Exp3 recall falls from 0.835 control to 0.682 at scale 0.25. This is a synthetic scale change, not a measured physical distance. |
| Dataset differs from live monitoring | Public and self-collected photographs do not establish performance in continuous household-camera scenes. Synthetic stress images are derived from existing photos, not independent camera data. |
| Earlier test exposure | Test photographs informed earlier project development, even though they are excluded from training in this final split. This is not a completely unseen benchmark for the whole project. |
| Annotation and duplicate uncertainty | Format checks cannot ensure complete/correct labels. Grouped visual hashes reduce leakage risk but may miss related photos; four large candidate buckets were skipped. |
| Synthetic-training limits | Pasted crops can retain background and visible boundaries. Background crops rely on complete source annotations. |
| Different transformed populations | Visibility filtering and transformations can change evaluated annotations. Size bins at 640 versus 960 contain different object populations. Stress improvements in a bin are not proof degradation helps. |
| Single-run evidence | One seed and fixed budget per configuration; no multi-seed confidence intervals. The shared recipe does not isolate augmentation's benefit. |
| No measured physical operating range | Pixels and synthetic brightness/rotation do not establish metres, lux limits or true 3D viewpoint robustness. |
| False boxes are not parent alerts | No labelled-empty evaluation scenes were available; empty-scene alarm rates are unavailable, not zero. Temporal alerts need separate evaluation. |
| Application validation remains | Live capture, depth-map age, latency, grouped-class spatial handling and alert behaviour need end-to-end checks. No detection does not prove safety. |

Evidence: [class results](../results/model_training/final_test/selected_model_classes.csv), [size results](../results/model_training/final_test/recall_by_pixel_size.csv), [controlled conditions](../results/model_training/final_test/controlled_condition_results.csv), and notebook Sections 8–10. The spatial/depth timing limitations in the UI guide are application observations, separate from these detector measurements.

## Future work and application handoff

1. Review difficult-class annotations and confusing/missed examples; collect representative live-camera data and independently labelled empty scenes.
2. Evaluate the same objects under recorded camera distance, lighting, placement and capture resolution; measure end-to-end delay and alert outcomes.
3. Repeat key configurations across seeds before treating small differences as conclusive.
4. Investigate classifier proposal crops, rejected true hazards and reclassification errors before further pipeline use.
5. Integrate `final_best.pt` with input 640 and confidence 0.30. Preserve model class names, keep Exp6 disabled, and review the legacy spatial mapping for grouped `small_object`, `sharp`, `medicine` and `pen` labels. A group-level prediction cannot recover the original fine-grained class.

The current `modules/spatial.py` risk weights use fourteen legacy classes; grouped labels are not fully specified there. The existing alias for `medicine_syrup` does not handle grouped `medicine`. This documentation update records the mismatch without changing application behaviour or choosing new risk weights.

## Reproduction and records

The root notebook is a compilation of separate sessions with preserved outputs, not proof of one uninterrupted Run All. Reproduction requires the raw dataset for training and the saved bundle for final evaluation. The bundle loader reloads historical checkpoints; retraining earlier cells does not automatically replace that bundle.

The final evaluation log records **Python 3.13.16, Ultralytics 8.4.198, Albumentations 2.0.8, Torch 2.11.0+cu130 and Tesla T4**. The allocated accelerator was T4 x2 but code used device 0. This differs from the local app's Python/dependency context; the root requirements file is preserved, not claimed to be an exact Kaggle lockfile.

Per-run histories and original `args.yaml` files are under [training_history](../results/model_training/training_history/). Arguments can contain historical Kaggle paths or framework-specific YAML tags and are evidence, not portable configs to load blindly. [Split membership](../results/model_training/split_membership.csv) retains exported filename, category, split and photo-group ID, without image data or source-path columns.

The baseline source session was stopped after completing baseline and Exp1; its partial Exp2 is excluded. Exp2–6 records use their completed standalone sessions. See the [artifact manifest](../configs/model_training/artifact_manifest.json) for session references, hashes and exact local records copied into this repo.

## Artifact locations and verification

| Artifact | Location / record |
|---|---|
| Baseline + Exp1 | [Kaggle baseline notebook, version 8](https://www.kaggle.com/code/oliviaee/cvdl-final-baseline-exp1) — manifest records version 8 and session 369424590 |
| Exp2 | [Kaggle Exp2](https://www.kaggle.com/code/oliviaee/cvdl-exp2) |
| Exp3 | [Kaggle Exp3](https://www.kaggle.com/code/oliviaee/cvdl-exp3) |
| Exp4 | [Kaggle Exp4](https://www.kaggle.com/code/tzehui/cvdl-exp4) |
| Exp5 | [Kaggle Exp5](https://www.kaggle.com/code/tzehui/cvdl-exp5) |
| Exp6 | [Kaggle Exp6](https://www.kaggle.com/code/tzehui/cvdl-final-exp6) |
| Final evaluation | [Kaggle Sections 7–10](https://www.kaggle.com/code/tzehui/cvdl-final-section-7-10) |
| Evaluation input | Private Kaggle Dataset `tzehui/cvdl-output`; bundle root `CVDL_Final_Evaluation_Bundle` |
| Selected checkpoint | Final evaluation output: `CVDL_Final_Evaluation/models/final_best.pt`; copy locally to `weights/final_best.pt` |
| Full failure records and images | Original final evaluation output, including `object_outcomes.csv`, `metrics/` and `evaluation_images/` |

Kaggle pages/datasets may require author-granted access. Page URLs can advance to newer versions; compare source-session records and SHA-256 checksums in the manifest rather than assuming the latest output is identical. Models and datasets are not uploaded by this change. Source image URLs/licences and earlier V2/V3 report citations are not fully supplied in the notebook and still need attribution before final submission.

To verify a downloaded model on Windows:

```powershell
Get-FileHash -Algorithm SHA256 .\weights\final_best.pt
```

Compare the result with the `final_best.pt` entry in the manifest. Existing `weights/`, runtime databases and secrets stay ignored. The executed notebook is around 38 MB, above GitHub's browser upload limit; push it using Git or distribute the executed file separately. Do not clear its saved results just to upload it.

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/): model family.
- [Ultralytics validation](https://docs.ultralytics.com/modes/val/): mAP and evaluator documentation.
- [Albumentations](https://albumentations.ai/docs/): image and bounding-box transformations. Current documentation may describe newer releases; recorded versions above govern this experiment.
- Internal evidence: notebook Sections 6–10 and the linked CSVs/training histories. Original public-image sources and earlier-project bibliographic details remain to be completed; no missing source attribution has been invented.
