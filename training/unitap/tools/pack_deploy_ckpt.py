"""Slim deployment checkpoints: EMA applied to the trainable modules, optimizer/metrics/EMA copy dropped.
Output keeps the {"args","modules"} layout, so tools/infer_pipeline.py loads it unchanged."""
import argparse, sys, torch
from pathlib import Path
WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "tools"))
import infer_pipeline as IP  # noqa: E402 (sets up sys.path for train.py)
from train import TrainableEMA, build_model, load_trainable_state_dict  # noqa: E402

OUT = WS / "UniTAP_deploy_ckpt"; R = WS / "experiments/runs"
HEADS = dict(stage1_head=R / "F1_stage1_all/out/checkpoints/best.pt", stage2_head=R / "F2_zoom_all/out/checkpoints/best.pt",
             contact_head=R / "F3_contact_hs_all/out/checkpoints/best.pt")
REFINER = R / "F4_refiner_all/pixref/best.pt"

def slim_head(src, dst):
    c = torch.load(src, map_location="cpu", weights_only=False); args = argparse.Namespace(**dict(c["args"])); args.device = "cpu"
    m = build_model(args, num_combo_classes=len(IP.LABELS["combo2id_6class"])); load_trainable_state_dict(m, c["modules"])
    if "ema" in c:
        e = TrainableEMA(model=m, decay=float(getattr(args, "ema_decay", 0.999))); e.load_state_dict(c["ema"], model=m); e.apply_shadow(m)
    mods = {n: {k: v.detach().cpu().contiguous() for k, v in mod.state_dict().items()} for n, mod in m.trainable_modules.items()}
    torch.save({"args": dict(c["args"]), "modules": mods}, dst)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for n, s in HEADS.items():
        slim_head(s, OUT / f"{n}.pt"); print(n, (OUT / f"{n}.pt").stat().st_size / 1e6, "MB")
    r = torch.load(REFINER, map_location="cpu", weights_only=False)
    torch.save({"args": r["args"], "model": r["model"]}, OUT / "refiner.pt"); print("refiner", (OUT / "refiner.pt").stat().st_size / 1e6, "MB")
