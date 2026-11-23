# TACTUS: Unifying Surgical Action Understanding in Time and Space

**TACTUS** (Tool, Action, Contact and Trajectory Understanding in Surgery) is a dataset and benchmark for following an instrument–tissue interaction from approach through completion. Each clip ties the identity of the instrument and action to the interaction's timing, the instrument's trajectory, and the tissue region involved.

**[Dataset on Hugging Face](https://huggingface.co/datasets/hding2455/surgical-action-affordance)** · **[Annotation guide](data/README.md)** · **[TactusNet](models/README.md)** · **[Evaluation protocol](evaluation/README.md)**

## Dataset

TACTUS contains **22,995 action clips** spanning approximately **2.9 million frames** from **141 videos**. It covers six tool–action pairs across five sources: Cholec80, HeiChole, CRCD, SurgVU, and a YouTube cholecystectomy collection.

| Tier | Clips | Videos | How the annotations are constructed |
| --- | ---: | ---: | --- |
| Gold | 15,668 | 101 | Human annotators label the tool, action, and time windows and place spatial prompts. Point tracking is followed by review and a final re-check. |
| Silver | 7,327 | 40 | Human annotators label the tool, action, and time windows. A model places spatial prompts for tracking on additional SurgVU clips. |

## What a clip describes

| Question | Annotation |
| --- | --- |
| **Which instrument acts?** | One of four tools: hook, grasper, scissors, or clipper. |
| **What does it do?** | One of four actions: dissect, grasp, clip, or cut. |
| **When does the interaction occur?** | An interaction window inside the full action window, which also includes approach and completion. |
| **How does the instrument move?** | Three instrument-specific landmarks tracked across the action window, with per-frame visibility. |
| **Where does it act on tissue?** | Four tissue-point tracks enclosing the affected region, also with per-frame visibility. |

For example, a grasper–grasp clip captures the approach, jaw closure, and retraction. The interaction window starts when the jaws begin to close and ends when the instrument stops moving after retraction. The tool landmarks describe the motion of the jaws, while the tissue points follow the region being grasped.

## Code release

This repository currently contains the dataset, model, and benchmark documentation. The TactusNet implementation and evaluation scripts will be added here.

## Authors

Lurui Wang¹*, Hao Ding¹*, Chenhao Yu², Chenyan Jing¹, Jiru Xu¹, Ruoqi Zuo¹, Liwei Mei¹, Chang Liu³, Lalithkumar Seenivasan¹, and Mathias Unberath¹†

¹ Johns Hopkins University · ² University of Virginia, Charlottesville · ³ University of Toronto

\* Equal contribution: [Lurui Wang](mailto:lwang255@jhu.edu) and [Hao Ding](mailto:hding2455@gmail.com).  
† Corresponding author: [Mathias Unberath](mailto:mathias@jhu.edu).
