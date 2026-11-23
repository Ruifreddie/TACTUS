#!/usr/bin/env python3
"""Evaluate a classification or interval checkpoint on one split with the trainer's own evaluate(). usage: --code --ckpt --root --split --out"""
import argparse, json, sys
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--code', required=True); ap.add_argument('--ckpt', required=True); ap.add_argument('--root', required=True)
ap.add_argument('--split', default='test'); ap.add_argument('--out', required=True); ap.add_argument('--batch-size', type=int, default=32)
c = ap.parse_args(); sys.path.insert(0, f'{c.code}/kpt_combo'); sys.path.insert(0, c.code)
import train as T
ck = torch.load(c.ckpt, map_location='cpu', weights_only=False); a = argparse.Namespace(**dict(ck['args']))
a.data_root = c.root; a.val_split_file = f'{c.root}/{c.split}.txt'; a.vjepa_cache_dir = f'{c.root}/vjepa_token_cache'; a.batch_size = c.batch_size; a.num_workers = 4
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); a.device = str(dev)
ds = T.SurgKptComboDataset(root=c.root, split='val', split_file=a.val_split_file, input_size=a.input_size, normalize=True, load_video=False,
                           combo_only_pure=bool(getattr(a, 'combo_only_pure', False)), tool_kpt_only=False)
# the evaluator reads tokens from cache/<split>; point its "val" at the requested split
class TC(T.VJEPATokenCache):
    def _path(self, split, sid): return super()._path(c.split, sid)
    def path_for(self, split, sid): return super().path_for(c.split, sid)
tc = TC(root=Path(a.vjepa_cache_dir), enabled=True, dtype_name=str(getattr(a, 'vjepa_cache_dtype', 'fp16')))
loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)
m = T.build_model(a, num_combo_classes=int(ds.num_combos)).to(dev); T.load_trainable_state_dict(m, ck['modules']); m.eval()
if 'ema' in ck:
    e = T.TrainableEMA(model=m, decay=float(getattr(a, 'ema_decay', 0.999))); e.load_state_dict(ck['ema'], model=m); e.apply_shadow(m)
r = T.evaluate(m, loader, dev, a, ds.combo_id_to_name, token_cache=tc)
metrics, extra = (r if isinstance(r, tuple) else (r, {}))
clean = lambda d: {k: (float(v) if hasattr(v, '__float__') else v) for k, v in d.items()}
out = {'epoch': ck.get('epoch'), 'split': c.split, 'n': len(ds), 'metrics': clean(metrics), 'per_combo': {k: clean(v) for k, v in extra.items() if isinstance(v, dict)}}
json.dump(out, open(c.out, 'w'), indent=1); print(json.dumps(out['metrics'])[:1500])
