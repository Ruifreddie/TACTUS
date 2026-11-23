# Reading the TACTUS annotations

The [Hugging Face dataset](https://huggingface.co/datasets/hding2455/surgical-action-affordance) contains an export from the annotation tool. Start with the index for the annotation tier you want. Each index maps an annotation ID to its source video, frame range, and associated files.

| Index file | Contents |
| --- | --- |
| `index_gold.json` | 15,668 Gold annotations from 101 videos |
| `index_silver.json` | 7,327 Silver annotations from 40 SurgVU videos |
| `index_all.json` | All 22,995 annotations from 141 videos |

## Follow one annotation

1. Pick an entry from an index. The `task` field identifies the source video. The `tool` and `action` fields identify the maneuver.
2. Open the file at `anno_dir`, a path relative to the dataset root. The export uses `affordance_range` for the paper's **action window** and `action_range` for its inner **interaction window**.
3. Read `points` for the frame-indexed instrument and tissue tracks. Each point records its ID, type, source-frame index, coordinates, and visibility. Track coordinates are in source-video pixels. The separate file at `prompt_dir` uses coordinates normalized by the image dimensions.
4. To align a track with the exported clip, subtract the index entry's `start` from its source-frame index. Both endpoints are inclusive, so a clip contains `end - start + 1` frames.

The six annotated pairs are hook–dissect, grasper–grasp, grasper–dissect, scissors–dissect, clipper–clip, and scissors–cut.

The Gold and Silver indexes describe annotation tiers. For comparisons with the paper, use the [video-level benchmark split](../evaluation/README.md).

## Source terms

The current dataset card describes an October 2026 annotation-tool export and labels it “Private, internal use.” Review the [dataset card](https://huggingface.co/datasets/hding2455/surgical-action-affordance/blob/main/README.md) and the terms of the original video source before reusing or redistributing clips.
