#!/usr/bin/env python3
"""Pretrained weights of the classification baselines, downloaded into $BL_WEIGHTS/cls/<model>/ when missing (idempotent).

  python -m baselines.cls.weights [--models tsn,tsm,videomae,xclip,endofm,mvitv2]

Every source is pinned (file hash, HF commit or Drive id) and the sha256 of what was fetched is written to
$BL_WEIGHTS/cls/MANIFEST.json, so a pod run can be checked against the local one.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

from baselines.common import config

ROOT = Path(os.environ.get("BL_WEIGHTS", str(config.WEIGHTS))) / "cls"

SOURCES = {
    # TSN: the TSM repo has no TSN checkpoint for Something-Something (only for Kinetics), so TSN starts from the
    # ImageNet ResNet-50 that the TSM repo itself loads (torchvision resnet50(pretrained=True) = IMAGENET1K_V1).
    "tsn": dict(kind="url", file="resnet50-0676ba61.pth", url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
                desc="torchvision ResNet-50 IMAGENET1K_V1 (ImageNet-1k supervised)"),
    "tsm": dict(kind="url", file="TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth",
                url="https://hanlab18.mit.edu/projects/tsm/models/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth",
                desc="TSM ResNet-50, shift_div 8, blockres, 16 segments, Something-Something V2 (ImageNet init), 45 epochs"),
    "videomae": dict(kind="hf", repo="MCG-NJU/videomae-base-finetuned-ssv2", revision="e3f23b7a9bcc6f6b0bbf44da097c0f94548c6ec5",
                     files=("config.json", "preprocessor_config.json", "pytorch_model.bin"),
                     desc="VideoMAE ViT-B/16, self-supervised on SSv2 then fine-tuned on SSv2 (16x224)"),
    "xclip": dict(kind="hf", repo="microsoft/xclip-base-patch16-16-frames", revision="c4bb0025a08ee79a5e74ef77e511c6f83f328d77",
                  files=("config.json", "preprocessor_config.json", "pytorch_model.bin", "merges.txt", "vocab.json",
                         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
                  desc="X-CLIP ViT-B/16, 16 frames: CLIP (WIT-400M) then fully supervised on Kinetics-400"),
    "endofm": dict(kind="gdrive", file="endo_fm.pth", id="1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl",
                   desc="Endo-FM ViT-B/16 TimeSformer, self-supervised on endoscopy videos (init SVT K400 SSL); "
                        "Google Drive link of github.com/med-air/Endo-FM README"),
    "mvitv2": dict(kind="url", file="mvit_v2_s-ae3be167.pth", url="https://download.pytorch.org/models/mvit_v2_s-ae3be167.pth",
                   desc="torchvision MViT_V2_S_Weights.KINETICS400_V1 (Kinetics-400 supervised, 16x224)"),
}


def _sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def _manifest_update(name, entry):
    m = ROOT / "MANIFEST.json"; d = json.loads(m.read_text()) if m.exists() else {}
    d[name] = entry; tmp = m.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=1)); os.replace(tmp, m)


class _lock:
    """Exclusive file lock on $BL_WEIGHTS/cls/<name>/.lock: the cls and ablate preparations run in parallel on one pod and must
    not download (or extract) the same checkpoint into the same .part file at the same time."""
    def __init__(self, name): self.p = ROOT / name; self.p.mkdir(parents=True, exist_ok=True)
    def __enter__(self):
        import fcntl
        self.f = open(self.p / ".lock", "w"); fcntl.flock(self.f, fcntl.LOCK_EX); return self
    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def path(name):
    """Local path of the weights of `name` (a file, or a directory for Hugging Face models); downloads when missing."""
    with _lock(name):
        return _path(name)


def _path(name):
    s = SOURCES[name]; d = ROOT / name; d.mkdir(parents=True, exist_ok=True)
    if s["kind"] == "hf":
        if not all((d / f).is_file() for f in s["files"]):
            from huggingface_hub import hf_hub_download
            for f in s["files"]:
                if not (d / f).is_file():
                    hf_hub_download(s["repo"], f, revision=s["revision"], local_dir=str(d))
            _manifest_update(name, {"repo": s["repo"], "revision": s["revision"], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "sha256": {f: _sha256(d / f) for f in s["files"] if f.endswith((".bin", ".safetensors"))}})
        return d
    p = d / s["file"]
    if not p.is_file():
        tmp = d / (s["file"] + ".part")
        if s["kind"] == "url":
            import torch
            prefix = s["file"].rsplit("-", 1)[-1].split(".")[0] if s["url"].startswith("https://download.pytorch.org") else None
            torch.hub.download_url_to_file(s["url"], str(tmp), hash_prefix=prefix, progress=True)
        elif s["kind"] == "gdrive":
            import gdown
            out = gdown.download(id=s["id"], output=str(tmp), quiet=False)
            if out is None or not tmp.is_file(): raise IOError(f"gdown could not fetch Drive file {s['id']} (quota or link change?)")
        else:
            raise ValueError(s["kind"])
        if tmp.stat().st_size < 1 << 20: raise IOError(f"{tmp} is only {tmp.stat().st_size} bytes: not a checkpoint")
        os.replace(tmp, p)
        _manifest_update(name, {k: v for k, v in s.items() if k in ("url", "id", "file")} |
                         {"sha256": _sha256(p), "bytes": p.stat().st_size, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    return p


def endofm_backbone():
    """The teacher backbone of the Endo-FM checkpoint (what Endo-FM's eval_finetune.py loads) as a plain state dict,
    extracted once with weights_only=True (the full file also holds student, motion branches, optimizer and args)."""
    dst = ROOT / "endofm" / "endo_fm_teacher_backbone.pth"
    if dst.is_file(): return dst                       # extracted before (or copied from the volume): no 2.3 GB download
    src = path("endofm")
    with _lock("endofm"):
        return _endofm_extract(src)


def _endofm_extract(src):
    dst = src.with_name("endo_fm_teacher_backbone.pth")
    if not dst.is_file():
        import argparse as _ap, codecs, numpy as np, torch
        core = getattr(np, "_core", None) or np.core                     # numpy 2 / numpy 1
        safe = [_ap.Namespace, codecs.encode, np.dtype, (core.multiarray.scalar, "numpy.core.multiarray.scalar")]
        safe += [getattr(np.dtypes, n) for n in ("Float64DType", "Float32DType", "Int64DType") if hasattr(np, "dtypes")]
        with torch.serialization.safe_globals(safe):
            ck = torch.load(src, map_location="cpu", weights_only=True)
        sd = {k[len("backbone."):]: v.clone() for k, v in ck["teacher"].items() if k.startswith("backbone.")}
        tmp = dst.with_suffix(".tmp"); torch.save(sd, tmp); os.replace(tmp, dst)
        _manifest_update("endofm_teacher_backbone", {"from": src.name, "pretrain_epoch": int(ck.get("epoch", -1)),
                                                     "n_tensors": len(sd), "sha256": _sha256(dst)})
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=",".join(SOURCES)); a = ap.parse_args(argv)
    for n in a.models.split(","):
        p = path(n); print(f"{n:9s} {p}")
        if n == "endofm": print(f"{'':9s} {endofm_backbone()}")
    print("manifest", ROOT / "MANIFEST.json")


if __name__ == "__main__":
    sys.exit(main())
