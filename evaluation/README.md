# TACTUS evaluation protocol

The benchmark evaluates recognition, temporal localization, instrument trajectories, and tissue regions on the same Gold action clips. Each video belongs to one partition. Gold-only, Silver-only, and Gold + Silver models use Gold validation clips for checkpoint selection and the same Gold test set for evaluation.

| Gold benchmark partition | Videos | Clips |
| --- | ---: | ---: |
| Training | 79 | 12,372 |
| Validation | 11 | 1,604 |
| Test | 11 | 1,523 |
| **Total** | **101** | **15,499** |

The Gold release contains 15,668 clips. The table gives the fixed benchmark counts reported in Supplement D.1, Table S9. The [pre-export indices](../training/pipeline/splits/gold_v2/) contain 12,443 training, 1,654 validation, and 1,541 test records and specify each video's partition. The same five files are available in [the Hugging Face mirror](https://huggingface.co/datasets/hding2455/TACTUS/tree/main/splits/gold_v2). The exact clip lists used for the paper are not included in this repository.

For a new export, the [exporter](../training/pipeline/export_clips.py) writes the successfully exported clip IDs to `all_train.txt`, `all_val.txt`, and `all_test.txt`. It records export failures in `export_report_<groups>.json`. The scorer reads the clip list from that export. [Gold release index](https://huggingface.co/datasets/hding2455/TACTUS/blob/main/index_gold.json)

## Metrics

Each clip supplies 16 sampled frames spanning its action window. Spatial errors use coordinates normalized to the video's fixed content crop and multiplied by 384 in both axes. The main localization comparison supplies ground-truth tool and action labels. The paper also reports a separate evaluation conditioned on predicted classes.

| Task | Scoring and aggregation (Supplement D.2) |
| --- | --- |
| Which / What | The scorer selects the highest-scoring tool–action pair and reads the tool and action from that pair (Tables 3 and 4a). Each clip has equal weight for tool, action, and pair accuracy. Pair mAcc averages the accuracies of the six true pairs equally. Table 4c instead uses the separate tool and action logits for Which and What. |
| When | tIoU compares continuous intervals on the action-window axis, with 0 at the first sampled frame and 1 at the last. Predictions are clipped to this range. Mean tIoU and recall weight each valid clip equally. R@0.3, R@0.5, and R@0.7 count clips whose tIoU meets or exceeds the threshold. |
| How | ADE averages visible tool-point errors within each clip, then averages the clip means equally. A clip with zero visible tool points contributes no ADE. The median, share above 32 px, and PCK pool all visible points. PCK@4, @8, @12, and @16 count errors strictly below their respective thresholds in pixels. |
| Where | Dice averages frame-level overlap over frames with at least two visible annotated contact points (21,070 frames in the reported test set). The target is the hull of the visible points. For point predictions, the region is the hull of all four predicted points. Hulls use a 6 px disc at each vertex and a 12 px line for a two-vertex hull. Mask methods use `contact_frame_dice` with binary masks on the 384 × 384 grid and save the per-frame scores. |

For the TactusNet analyses in Supplement Table S10, the inference script reports ADE pooled over points in its `refined` summary. The contact evaluation script reports hull IoU, Dice grouped by the number of visible points, and Hungarian-matched contact-point error. [Tool-point inference](../training/unitap/tools/infer_pipeline.py) · [Contact evaluation](../training/unitap/tools/eval_contact_stage.py)

## Running the evaluation

Set `BL_EXPORT` to the directory containing the exported split lists and per-clip arrays. Save one prediction file per method, task family, and split. The scorer requires a prediction for every clip in the selected export list. [Prediction format and baseline examples](../baselines/README.md) · [Scorer](../baselines/common/score.py)

```bash
# score one method on the Gold test set
python -m baselines.common.score --pred OUT/my_method/pred_test.npz --out OUT/my_method/metrics_test.json

# compare two methods: paired bootstrap of A - B over the test videos
python -m baselines.common.score --pred A.npz --compare B.npz
```

`--out` writes the full report. It includes the full split (`all`), clips outside SurgVU (`non_surgvu`), and each represented source and true pair. Accuracies, recalls, and PCK are fractions. For `all` and `non_surgvu`, the `ci95` block covers metrics with an accumulated numerator and denominator. Per-source and per-pair reports, pair mAcc, the pooled median, and the optional fallback rate give point estimates.

For `--compare`, use two prediction files for the same split and task family. For each shared metric, the paired bootstrap draws from the videos contributing to both files' scores (`n_units`). It resamples whole videos with replacement 2,000 times and scores both files on each draw. A video drawn twice contributes twice with all its eligible observations. For ADE this gives each eligible clip equal weight within the draw. The reported difference is A minus B on the original split. The 95% interval uses the 2.5th and 97.5th percentiles of the resampled differences.
