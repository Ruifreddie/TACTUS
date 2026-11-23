"""Pretrained BACKBONE weights of the keypoint baselines, downloaded into $BL_WEIGHTS/kpt/ when missing (idempotent).

Initialization rule of TACTUS: every baseline starts from public pretrained backbone weights only; every task head and
decoder is trained from scratch on TACTUS. No COCO pose or COCO detection checkpoint is used.

  hrnet    HRNet-W32, ImageNet-1k classification: hrnet_w32-36af842e.pth, the file the official HRNet / DarkPose COCO configs
           initialize from (MODEL.PRETRAINED), from the OpenMMLab mirror; "36af842e" is its MD5 prefix
  vitpose  ViT-B/16, MAE self-supervised on ImageNet-1k: mae_pretrain_vit_base.pth, the official MAE release that the
           ViTPose README points to (encoder only; no hash in its name, so its sha256 is pinned here)
  dsta     ResNet-50, ImageNet-1k supervised: torchvision ResNet50_Weights.IMAGENET1K_V1 (resnet50-0676ba61.pth)

Every file is hash-checked and read with torch.load(weights_only=True). torchvision downloads go to
$BL_WEIGHTS/kpt/torch_hub (torch.hub dir), never to the repo or the system disk.

  python -m baselines.kpt.weights            # download all, print paths
"""
import contextlib, hashlib, shutil, sys, urllib.request
from pathlib import Path

from baselines.common import config

ROOT = Path(config.WEIGHTS) / "kpt"
TORCH_HUB = ROOT / "torch_hub"

# name -> (url, (hash algorithm, full hex or prefix), what it is)
FILES = {
    "hrnet_w32_imagenet": ("https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth", ("md5", "36af842e"),
                           "HRNet-W32 ImageNet-1k classification (hrnet_w32-36af842e.pth, the init file of the official HRNet/DarkPose configs)"),
    "mae_vit_base": ("https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
                     ("sha256", "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"),   # 343,249,461 bytes, 2026-11-05
                     "MAE ViT-B/16, self-supervised on ImageNet-1k, encoder only (official MAE release mae_pretrain_vit_base.pth)"),
}


def _digest(path, algo, chunk=1 << 20):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()


def fetch(name):
    """Local path of a checkpoint file, downloading and hash-checking it first if needed."""
    url, (algo, want), _ = FILES[name]
    d = ROOT / name; d.mkdir(parents=True, exist_ok=True); p = d / url.rsplit("/", 1)[-1]; ok = d / ".hash_ok"
    if p.exists() and ok.exists(): return p
    if not p.exists():
        tmp = p.with_suffix(p.suffix + ".part")
        print(f"[weights] downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f: shutil.copyfileobj(r, f, 1 << 22)
        tmp.replace(p)
    got = _digest(p, algo)
    if not got.startswith(want):
        p.unlink(); raise RuntimeError(f"{name}: {algo} {got[:16]} does not match the pinned {want[:16]}; file removed")
    ok.write_text(f"{algo} {got}\nsha256 {_digest(p, 'sha256') if algo != 'sha256' else got}\n")
    return p


@contextlib.contextmanager
def torch_hub_dir():
    """torchvision weights download into $BL_WEIGHTS/kpt/torch_hub/checkpoints (hash-checked by torchvision)."""
    import torch
    old = torch.hub.get_dir(); TORCH_HUB.mkdir(parents=True, exist_ok=True); torch.hub.set_dir(str(TORCH_HUB))
    try: yield
    finally: torch.hub.set_dir(old)


def tv_resnet50_state():
    """torchvision ResNet50_Weights.IMAGENET1K_V1 state_dict (torchvision key names)."""
    from torchvision.models import ResNet50_Weights
    with torch_hub_dir():
        return ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)


def load_state(name):
    """state_dict of a downloaded checkpoint (unwraps {'state_dict': ...} and the MAE release's {'model': ...})."""
    import torch
    s = torch.load(fetch(name), map_location="cpu", weights_only=True)
    if isinstance(s, dict):
        for k in ("state_dict", "model"):
            if k in s and isinstance(s[k], dict): return s[k]
    return s


def describe(name):
    """Provenance of a backbone checkpoint, for info.json and the paper's Backbone column."""
    if name == "tv_resnet50":
        from torchvision.models import ResNet50_Weights
        w = ResNet50_Weights.IMAGENET1K_V1
        return {"name": "torchvision ResNet50_Weights.IMAGENET1K_V1", "file": w.url.rsplit("/", 1)[-1], "url": w.url,
                "what": "ResNet-50, ImageNet-1k supervised (top-1 76.13)", "path": str(TORCH_HUB / "checkpoints" / w.url.rsplit("/", 1)[-1])}
    url, (algo, want), what = FILES[name]; f = url.rsplit("/", 1)[-1]; ok = ROOT / name / ".hash_ok"
    return {"name": f, "url": url, "what": what, "hash": ok.read_text().split() if ok.exists() else [algo, want], "path": str(ROOT / name / f)}


def fetch_all():
    out = {n: str(fetch(n)) for n in FILES}; tv_resnet50_state(); out["tv_resnet50"] = describe("tv_resnet50")["path"]
    return out


if __name__ == "__main__":
    for k, v in fetch_all().items(): print(k, v)
    sys.exit(0)
