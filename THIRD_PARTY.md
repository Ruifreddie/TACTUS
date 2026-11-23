# Third-party code

Code written for TACTUS is under the [MIT License](LICENSE). The directories below hold code from other projects. Each keeps its upstream license, and that license governs its use.

| Directory | Project | License | Details |
| --- | --- | --- | --- |
| `training/unitap/third_party/t2b/vjepa2/` | [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (Meta) | MIT, with Apache-2.0 parts (`LICENSE`, `APACHE-LICENSE` in the directory) | Model definitions used to build the V-JEPA 2.1 encoder. `training/repro/shared/vjepa2` links to the same copy. |
| `baselines/cls/vendor/tsm/` | [TSM](https://github.com/mit-han-lab/temporal-shift-module) | MIT | [`baselines/cls/THIRD_PARTY.md`](baselines/cls/THIRD_PARTY.md) |
| `baselines/cls/vendor/endofm/` | [Endo-FM](https://github.com/med-air/Endo-FM) | Apache-2.0 | [`baselines/cls/THIRD_PARTY.md`](baselines/cls/THIRD_PARTY.md) |
| `baselines/when/third_party/actionformer/` | [ActionFormer](https://github.com/happyharrycn/actionformer_release) | MIT | [`baselines/when/THIRD_PARTY.md`](baselines/when/THIRD_PARTY.md) |
| `baselines/when/third_party/tridet/` | [TriDet](https://github.com/dingfengshi/TriDet) | MIT | [`baselines/when/THIRD_PARTY.md`](baselines/when/THIRD_PARTY.md) |
| `baselines/when/third_party/qd_detr/` | [QD-DETR](https://github.com/wjun0830/QD-DETR) | MIT, with Apache-2.0 file headers | [`baselines/when/THIRD_PARTY.md`](baselines/when/THIRD_PARTY.md) |
| `baselines/when/third_party/ms_tcn/` | [MS-TCN](https://github.com/yabufarha/ms-tcn) | MIT with Commons Clause (the code may not be sold) | [`baselines/when/THIRD_PARTY.md`](baselines/when/THIRD_PARTY.md) |
| `baselines/kpt/third_party/hrnet/` | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) | MIT | [`baselines/kpt/THIRD_PARTY.md`](baselines/kpt/THIRD_PARTY.md) |
| `baselines/kpt/third_party/dsta/` | [DSTA](https://github.com/zgspose/DSTA), with parts of mmpose and timm | Apache-2.0 | [`baselines/kpt/THIRD_PARTY.md`](baselines/kpt/THIRD_PARTY.md) |

The per-family files also list code that was re-implemented from a published method instead of copied, the commit each port follows, and the pretrained weights each baseline downloads with their licenses. [`baselines/where/THIRD_PARTY.md`](baselines/where/THIRD_PARTY.md) covers SegFormer, which is used as a library.

Pretrained encoders are downloaded at run time and are not part of this repository. Their licenses differ from model to model; check the model card before use.

The directories `training/unitap/third_party/t2b/kpt_combo`, `training/unitap/third_party/affordance_h613`, and `training/unitap/third_party/vitpose_dataset_tools` are earlier code of the TACTUS authors, kept at the paths the training scripts expect. They are covered by this repository's MIT License.
