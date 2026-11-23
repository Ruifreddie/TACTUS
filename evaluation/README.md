# TACTUS evaluation protocol

The benchmark evaluates recognition, temporal localization, instrument trajectories, and tissue-region localization on the same action clips. The main comparison trains on Gold and uses a video-level split: every video belongs to one partition.

| Gold benchmark partition | Videos | Clips |
| --- | ---: | ---: |
| Training | 79 | 12,372 |
| Validation | 11 | 1,604 |
| Test | 11 | 1,523 |
| **Total** | **101** | **15,499** |

The [Gold release index](https://huggingface.co/datasets/hding2455/surgical-action-affordance/blob/main/index_gold.json) contains 15,668 clips. The counts above describe the fixed subset used for the reported benchmark. Checkpoints are selected on the validation set. Experiments comparing Gold-only, Silver-only, and Gold + Silver training use the Gold test set for evaluation.

| Task | Main metrics |
| --- | --- |
| Which / What | Top-1 tool, action, and pair accuracy; mean accuracy across the six pairs. |
| When | Temporal intersection over union (tIoU) and recall at tIoU ≥ 0.7. |
| How | Average displacement error of visible instrument landmarks in pixels at 384 × 384, and the fraction of visible point predictions with error above 32 pixels. |
| Where | Dice overlap with the human-annotated tissue region. For keypoint methods, the predicted region is formed from the convex hull of the contact points. |

For the main localization comparison, models receive the ground-truth tool and action as inputs. The paper separately evaluates localization conditioned on predicted classes.

## Running the evaluation

The metric code is [`baselines/common/`](../baselines/common): `metrics.py` defines the metrics, `preds.py` the prediction file, and `score.py` scores a file. TactusNet and every baseline are scored through the same script.

```bash
# score one method on the Gold test set
python -m baselines.common.score --pred OUT/my_method/pred_test.npz --out OUT/my_method/metrics_test.json

# compare two methods: paired bootstrap of A - B over the test videos
python -m baselines.common.score --pred A.npz --compare B.npz
```

`score.py` reports each metric on all test clips, per source, and per tool–action pair, with a 95% bootstrap confidence interval over test videos. It reads the exported clips from `BL_EXPORT` (see [`baselines/README.md`](../baselines/README.md), which also specifies the prediction file and walks through adding a method).

The video-level split is in [`training/pipeline/splits/gold_v2/`](../training/pipeline/splits/gold_v2): `split_videos.csv` assigns each Gold video to a partition, and `index_gold_{train,val,test}.json` list the annotations of each partition in the format of the dataset's index files. The export step of the [training pipeline](../training/README.md) turns these lists into the per-clip arrays that training and scoring read.
