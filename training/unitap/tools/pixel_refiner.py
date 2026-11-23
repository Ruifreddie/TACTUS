"""Stage-2 pixel-crop refiner prototype (E008 candidate).

Motivation: the 24x24 V-JEPA tokens hold essentially no sub-patch position
(single-token probe 5.1 px vs patch-centre baseline 6.1 px @384, 3x3 context no better), and the stage-1
model's median error (6.9 px) sits at that wall.  Oracle: moving every point that is already within 16 px
down to 3 px would take val ADE 11.7 -> 8.9 and PCK@12 0.72 -> 0.80.  This module refines each stage-1
point from a native-resolution RGB crop (896x665 frames in data/case32/v1/video_16_npy, 2.33x finer than
the 384 grid) with a small CNN that predicts a heatmap over the crop.

Inputs per sample: 128x128 native crop centred on the stage-1 point, plus conditioning channels:
Gaussian blobs of the three stage-1 points (target slot first), tool one-hot, target role one-hot.
Training uses GT points displaced by error configurations sampled from an empirical bank of stage-1 val
errors (experiments/analysis/error_bank_E007j_val_384px.npy) so the crop-centre distribution matches
inference.  Evaluation crops around the real stage-1 predictions from a dump directory.

Usage:
  python tools/pixel_refiner.py selftest
  python tools/pixel_refiner.py train --out experiments/runs/E008/pixref --epochs 20
  python tools/pixel_refiner.py eval --ckpt experiments/runs/E008/pixref/best.pt --dump experiments/analysis/pred_E007j_best_val
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = 'data/case32/v1'
S384 = 384.0                     # native frame size varies per clip (W 895/896, H 613..720) -> taken from meta W/H per clip
CROP = 128                       # native px
HM = 32                          # heatmap cells per side (4 native px per cell)
COND_SIGMA = 6.0                 # native px, conditioning blobs
TARGET_SIGMA = 1.0               # cells, heatmap CE target
TOOLS = ['hook', 'grasper', 'scissors', 'clipper']
NAT2P384_APPROX = 0.5            # rough native px -> 384 px factor used only to scale the L1 loss term

def wh(m):
    return np.array([float(m['W']), float(m['H'])], dtype=np.float32)


# ----------------------------------------------------------------------------- data helpers
def read_clip_frames(sid):
    """Read [16,H,W,3] uint8 fully into memory and drop the file from the page cache."""
    path = f'{ROOT}/video_16_npy/{sid}.npy'
    with open(path, 'rb') as f:
        arr = np.load(f)
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
    return arr


def crop_at(frame, cx, cy):
    """128x128 crop centred at native (cx, cy); zero-padded. Returns crop uint8 [C,C,3] and origin (x0, y0)."""
    x0 = int(round(cx)) - CROP // 2; y0 = int(round(cy)) - CROP // 2
    out = np.zeros((CROP, CROP, 3), dtype=np.uint8)
    xs, ys = max(x0, 0), max(y0, 0); xe, ye = min(x0 + CROP, frame.shape[1]), min(y0 + CROP, frame.shape[0])
    if xe > xs and ye > ys:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = frame[ys:ye, xs:xe]
    return out, (x0, y0)


def cond_channels(pts_rel, tool_id, role):
    """pts_rel: [3,2] stage-1 points in crop px (target slot first). Returns [3+4+2, C, C] float32."""
    yy, xx = np.mgrid[0:CROP, 0:CROP].astype(np.float32)
    ch = []
    for p in pts_rel:
        ch.append(np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * COND_SIGMA ** 2)))
    for i in range(4):
        ch.append(np.full((CROP, CROP), 1.0 if i == tool_id else 0.0, dtype=np.float32))
    for i in range(2):
        ch.append(np.full((CROP, CROP), 1.0 if i == role else 0.0, dtype=np.float32))
    return np.stack(ch).astype(np.float32)


def make_sample(frame, pts_nat, k, tool_id, role, gt_nat=None):
    """Build one refiner sample around stage-1 point k. pts_nat [3,2] native px. Returns dict of numpy arrays."""
    cx, cy = pts_nat[k]
    crop, (x0, y0) = crop_at(frame, cx, cy)
    order = [k] + [j for j in range(3) if j != k]
    rel = pts_nat[order] - np.array([x0, y0], dtype=np.float32)
    x = np.concatenate([crop.transpose(2, 0, 1).astype(np.float32) / 255.0, cond_channels(rel, tool_id, role)], 0)
    s = dict(x=x, origin=np.array([x0, y0], dtype=np.float32), s1=np.array(pts_nat[k], dtype=np.float32))
    if gt_nat is not None:
        g = np.asarray(gt_nat, dtype=np.float32) - np.array([x0, y0], dtype=np.float32)
        s['gt_rel'] = g
        s['inside'] = np.float32(1.0 if (0 <= g[0] < CROP and 0 <= g[1] < CROP) else 0.0)
    return s


class TrainClips(torch.utils.data.Dataset):
    """One item = one clip: all visible GT points of all 16 frames, crops centred on GT + sampled stage-1 error."""

    def __init__(self, sids, meta, bank, jitter_scale=1.0, max_per_clip=None, seed=0):
        self.sids, self.meta, self.bank, self.js, self.maxn = sids, meta, bank, jitter_scale, max_per_clip
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        sid = self.sids[i]; m = self.meta[sid]; WH = wh(m)
        frames = read_clip_frames(sid); assert frames.shape[2] == int(WH[0]) and frames.shape[1] == int(WH[1]), (sid, frames.shape, WH)
        gt = np.load(f'{ROOT}/gt_tool_xy/{sid}.npy').astype(np.float32) * WH                  # [16,3,2] native px
        vis = np.load(f'{ROOT}/gt_tool_vis/{sid}.npy').astype(np.float32) > 0.5
        roles = list(m['pt_role']); tool_id = TOOLS.index(m['tool'])
        xs, gts, ins, s1s, orgs = [], [], [], [], []
        idx = [(t, k) for t in range(16) for k in range(3) if vis[t, k]]
        if self.maxn and len(idx) > self.maxn:
            idx = random.sample(idx, self.maxn)
        for t, k in idx:
            e = self.bank[np.random.randint(len(self.bank))] * self.js                          # [3,2] @384 px
            pts = gt[t] + e * (WH / S384)                                                       # native px
            # invisible points have no reliable GT: replace by target point + a random offset so blobs stay plausible
            for j in range(3):
                if not vis[t, j]:
                    pts[j] = pts[k] + np.random.randn(2) * 25.0
            s = make_sample(frames[t], pts.astype(np.float32), k, tool_id, roles[k], gt[t, k])
            xs.append(s['x']); gts.append(s['gt_rel']); ins.append(s['inside']); s1s.append(s['s1']); orgs.append(s['origin'])
        if not xs:
            return None
        return dict(x=torch.from_numpy(np.stack(xs)), gt_rel=torch.from_numpy(np.stack(gts)), inside=torch.from_numpy(np.stack(ins)),
                    s1=torch.from_numpy(np.stack(s1s)), origin=torch.from_numpy(np.stack(orgs)))


def collate_clips(items):
    items = [it for it in items if it is not None]
    if not items:
        return None
    return {k: torch.cat([it[k] for it in items], 0) for k in items[0]}


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sk = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sk is None else self.sk(x)))


class PixelRefiner(nn.Module):
    def __init__(self, cin=12, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.s1 = Block(w, w, 1)              # 128
        self.s2 = Block(w, 2 * w, 2)          # 64
        self.s3 = Block(2 * w, 4 * w, 2)      # 32
        self.s4 = Block(4 * w, 8 * w, 2)      # 16
        self.up = nn.Sequential(nn.Conv2d(8 * w, 4 * w, 3, 1, 1, bias=False), nn.BatchNorm2d(4 * w), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(Block(8 * w, 4 * w, 1), nn.Conv2d(4 * w, 1, 1))
        self.trust = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8 * w, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        yy, xx = torch.meshgrid(torch.arange(HM), torch.arange(HM), indexing='ij')
        self.register_buffer('grid', torch.stack([xx, yy], -1).float().reshape(-1, 2) * (CROP / HM) + (CROP / HM) / 2)  # cell centres, crop px

    def forward(self, x):
        f1 = self.s1(self.stem(x)); f2 = self.s2(f1); f3 = self.s3(f2); f4 = self.s4(f3)
        u = F.interpolate(self.up(f4), scale_factor=2, mode='bilinear', align_corners=False)
        logits = self.dec(torch.cat([u, f3], 1)).flatten(1)          # [B, HM*HM]
        trust = self.trust(f4).squeeze(1)
        return logits, trust

    def decode(self, logits, local=2):
        """Soft-argmax over the full map and a local soft-argmax (window radius `local` cells around the argmax)."""
        p = torch.softmax(logits, -1)
        full = p @ self.grid
        am = logits.argmax(-1); ay, ax = am // HM, am % HM
        yy = torch.arange(HM, device=logits.device).view(1, HM, 1).expand(-1, HM, HM); xx = torch.arange(HM, device=logits.device).view(1, 1, HM).expand(-1, HM, HM)
        mask = ((yy - ay.view(-1, 1, 1)).abs() <= local) & ((xx - ax.view(-1, 1, 1)).abs() <= local)
        pl = torch.where(mask.flatten(1), p, torch.zeros_like(p)); pl = pl / pl.sum(-1, keepdim=True).clamp_min(1e-9)
        loc = pl @ self.grid
        return full, loc


def gaussian_target(gt_rel):
    """gt_rel [B,2] crop px -> [B, HM*HM] normalized Gaussian over cells."""
    g = gt_rel / (CROP / HM) - 0.5
    yy, xx = torch.meshgrid(torch.arange(HM, device=gt_rel.device), torch.arange(HM, device=gt_rel.device), indexing='ij')
    d2 = (xx.float().reshape(1, -1) - g[:, 0:1]) ** 2 + (yy.float().reshape(1, -1) - g[:, 1:2]) ** 2
    t = torch.exp(-d2 / (2 * TARGET_SIGMA ** 2))
    return t / t.sum(-1, keepdim=True).clamp_min(1e-9)


def losses(model, batch, dev):
    x = batch['x'].to(dev, non_blocking=True); gt = batch['gt_rel'].to(dev); inside = batch['inside'].to(dev)
    logits, trust = model(x)
    full, loc = model.decode(logits)
    tgt = gaussian_target(gt)
    ce = -(tgt * F.log_softmax(logits, -1)).sum(-1)
    l1 = (full - gt).abs().sum(-1) * NAT2P384_APPROX                 # in ~384 px
    s1_rel = torch.full_like(gt, CROP / 2)                            # stage-1 point sits at the crop centre
    e_ref = (loc - gt).norm(dim=-1); e_s1 = (s1_rel - gt).norm(dim=-1)
    trust_tgt = ((e_ref < e_s1) & (inside > 0.5)).float()
    bce = F.binary_cross_entropy_with_logits(trust, trust_tgt)
    n_in = inside.sum().clamp_min(1.0)
    loss = (ce * inside).sum() / n_in + 0.05 * (l1 * inside).sum() / n_in + 0.2 * bce
    return loss, dict(ce=((ce * inside).sum() / n_in).item(), l1=((l1 * inside).sum() / n_in).item(), bce=bce.item(), inside=inside.mean().item())


# ----------------------------------------------------------------------------- train / eval
def meta_by_sid():
    return {m['sid']: m for m in json.load(open(f'{ROOT}/meta.json'))}


def train(a):
    dev = torch.device(a.device); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    meta = meta_by_sid(); bank = np.load(a.bank).astype(np.float32)
    sids = [l.strip() for l in open(a.train_list or f'{ROOT}/train.txt') if l.strip()]
    if a.max_clips: sids = sids[:a.max_clips]
    ds = TrainClips(sids, meta, bank, jitter_scale=a.jitter_scale, max_per_clip=a.max_per_clip, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.clips_per_batch, shuffle=True, num_workers=a.workers, collate_fn=collate_clips, persistent_workers=a.workers > 0, drop_last=False)
    model = PixelRefiner(w=a.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
    os.makedirs(a.out, exist_ok=True); best = 1e9; log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); agg = {}; nb = 0
        for batch in dl:
            if batch is None: continue
            # sub-batch to bound memory
            B = batch['x'].shape[0]
            for s in range(0, B, a.max_crops):
                sub = {k: v[s:s + a.max_crops] for k, v in batch.items()}
                loss, st = losses(model, sub, dev)
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
                for k, v in st.items(): agg[k] = agg.get(k, 0.0) + v
                agg['loss'] = agg.get('loss', 0.0) + float(loss); nb += 1
        tr = {k: v / max(nb, 1) for k, v in agg.items()}
        ev = evaluate(model, a.val_dump, meta, dev, max_clips=a.eval_clips) if a.val_dump else {}
        rec = dict(epoch=ep + 1, train=tr, val=ev, sec=time.time() - t0); log.write(json.dumps(rec) + '\n'); log.flush()
        print(f"[ep {ep + 1:03d}] loss {tr.get('loss', 0):.3f} ce {tr.get('ce', 0):.3f} l1 {tr.get('l1', 0):.2f} bce {tr.get('bce', 0):.3f} | val: " +
              (' '.join(f'{k}={v:.3f}' for k, v in ev.items() if isinstance(v, float)) if ev else '-') + f" | {time.time() - t0:.0f}s", flush=True)
        score = ev.get('ADE_gated', tr.get('loss', 1e9))
        torch.save(dict(model=model.state_dict(), args=vars(a), epoch=ep + 1, val=ev), os.path.join(a.out, 'last.pt'))
        if score < best:
            best = score; torch.save(dict(model=model.state_dict(), args=vars(a), epoch=ep + 1, val=ev), os.path.join(a.out, 'best.pt'))


@torch.no_grad()
def evaluate(model, dump, meta, dev, max_clips=None, save=None, batch=256, save_points=None):
    """Crop around the real stage-1 predictions of a dump; report raw / refined / gated ADE + PCK @384."""
    model.eval(); files = sorted(glob.glob(os.path.join(dump, '*.npz')))
    if max_clips: files = files[:max_clips]
    E_raw, E_ref, E_loc, E_gate, per_clip = [], [], [], [], {}
    PTS = {k: [] for k in ('e_raw', 'e_loc', 'e_full', 'disp', 'peak', 'ent', 'trust', 'tool', 'role', 'nvis', 'sid', 't', 'k')}
    for f in files:
        sid = os.path.basename(f)[:-4]; m = meta[sid]; z = np.load(f); WH = wh(m); P384 = S384 / WH
        pred = z['pred_xy'].astype(np.float32) * WH; gt = z['gt_xy'].astype(np.float32) * WH; vis = z['gt_vis'] > 0.5
        frames = read_clip_frames(sid); roles = list(m['pt_role']); tool_id = TOOLS.index(m['tool'])
        xs, orgs, gts, s1s, ids = [], [], [], [], []
        for t in range(16):
            for k in range(3):
                if not vis[t, k]: continue
                s = make_sample(frames[t], pred[t], k, tool_id, roles[k], gt[t, k])
                xs.append(s['x']); orgs.append(s['origin']); gts.append(gt[t, k]); s1s.append(pred[t, k]); ids.append((t, k, int(vis[t].sum())))
        if not xs: continue
        outs_full, outs_loc, outs_tr, outs_pk, outs_ent = [], [], [], [], []
        for s in range(0, len(xs), batch):
            x = torch.from_numpy(np.stack(xs[s:s + batch])).to(dev)
            logits, trust = model(x); full, loc = model.decode(logits); pr = torch.softmax(logits, -1)
            outs_full.append(full.cpu().numpy()); outs_loc.append(loc.cpu().numpy()); outs_tr.append(torch.sigmoid(trust).cpu().numpy())
            outs_pk.append(pr.max(-1).values.cpu().numpy()); outs_ent.append(-(pr * torch.log(pr.clamp_min(1e-9))).sum(-1).cpu().numpy())
        full = np.concatenate(outs_full) + np.stack(orgs); loc = np.concatenate(outs_loc) + np.stack(orgs); tr = np.concatenate(outs_tr); pk = np.concatenate(outs_pk); ent = np.concatenate(outs_ent)
        gts = np.stack(gts); s1s = np.stack(s1s)
        e_raw = np.linalg.norm((s1s - gts) * P384, axis=-1); e_full = np.linalg.norm((full - gts) * P384, axis=-1); e_loc = np.linalg.norm((loc - gts) * P384, axis=-1)
        gated = np.where(tr[:, None] > 0.5, loc, s1s); e_gate = np.linalg.norm((gated - gts) * P384, axis=-1)
        E_raw.append(e_raw); E_ref.append(e_full); E_loc.append(e_loc); E_gate.append(e_gate); per_clip[sid] = dict(raw=float(e_raw.mean()), gated=float(e_gate.mean()))
        if save_points is not None:
            disp = np.linalg.norm((loc - s1s) * P384, axis=-1)
            for key, val in [('e_raw', e_raw), ('e_loc', e_loc), ('e_full', e_full), ('disp', disp), ('peak', pk), ('ent', ent), ('trust', tr), ('tool', np.full(len(e_raw), tool_id)),
                             ('role', np.array([roles[k] for _, k, _ in ids])), ('nvis', np.array([n for _, _, n in ids])), ('sid', np.array([sid] * len(e_raw))), ('t', np.array([t for t, _, _ in ids])), ('k', np.array([k for _, k, _ in ids]))]:
                PTS[key].append(np.asarray(val))
    r = {}
    for name, E in [('raw', E_raw), ('full', E_ref), ('loc', E_loc), ('gated', E_gate)]:
        e = np.concatenate(E); r[f'ADE_{name}'] = float(e.mean()); r[f'med_{name}'] = float(np.median(e))
        for th in (4, 8, 12, 16): r[f'PCK{th}_{name}'] = float((e < th).mean())
        r[f'gt32_{name}'] = float((e > 32).mean())
    if save:
        json.dump(dict(summary=r, per_clip=per_clip), open(save, 'w'), indent=1)
    if save_points is not None:
        np.savez(save_points, **{k: np.concatenate(v) for k, v in PTS.items()})
    return r


def selftest():
    meta = meta_by_sid(); sid = [l.strip() for l in open(f'{ROOT}/val.txt') if l.strip()][0]
    frames = read_clip_frames(sid); gt = np.load(f'{ROOT}/gt_tool_xy/{sid}.npy').astype(np.float32) * wh(meta[sid])
    s = make_sample(frames[0], gt[0], 0, 0, 0, gt[0, 0])
    assert s['x'].shape == (12, CROP, CROP), s['x'].shape
    assert abs(s['gt_rel'][0] - CROP / 2) <= 0.5 and abs(s['gt_rel'][1] - CROP / 2) <= 0.5, s['gt_rel']
    m = PixelRefiner(); logits, tr = m(torch.from_numpy(s['x'])[None]); full, loc = m.decode(logits)
    assert logits.shape == (1, HM * HM) and full.shape == (1, 2)
    tgt = gaussian_target(torch.tensor([[CROP / 2, CROP / 2]])); c = tgt.argmax().item(); assert (c // HM, c % HM) in [(15, 15), (16, 16), (15, 16), (16, 15)], c
    # decoding sanity: put a peak at the GT cell -> decode returns ~gt
    lg = torch.log(tgt + 1e-9) * 3; full, loc = m.decode(lg); assert (full - CROP / 2).abs().max() < 2.0 and (loc - CROP / 2).abs().max() < 2.0, (full, loc)
    bank = np.load('experiments/analysis/error_bank_E007j_val_384px.npy')
    ds = TrainClips([sid], meta, bank); it = ds[0]; assert it['x'].shape[1:] == (12, CROP, CROP); print('train item crops', it['x'].shape[0], 'inside frac', float(it['inside'].mean()))
    b = collate_clips([it]); loss, st = losses(m, b, torch.device('cpu')); print('loss', float(loss), st)
    print('selftest OK', sid)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('selftest')
    p = sp.add_parser('train'); p.add_argument('--out', required=True); p.add_argument('--epochs', type=int, default=20); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--width', type=int, default=32); p.add_argument('--clips-per-batch', type=int, default=4); p.add_argument('--max-crops', type=int, default=256)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bank', default='experiments/analysis/error_bank_E007j_val_384px.npy'); p.add_argument('--jitter-scale', type=float, default=1.0)
    p.add_argument('--max-clips', type=int, default=0); p.add_argument('--max-per-clip', type=int, default=0)
    p.add_argument('--val-dump', default='experiments/analysis/pred_E007j_best_val'); p.add_argument('--eval-clips', type=int, default=0) ; p.add_argument('--train-list', default=None)
    p = sp.add_parser('eval'); p.add_argument('--ckpt', required=True); p.add_argument('--dump', required=True); p.add_argument('--device', default='cuda'); p.add_argument('--save', default=None); p.add_argument('--max-clips', type=int, default=0); p.add_argument('--save-points', default=None)
    a = ap.parse_args()
    if a.cmd == 'selftest': selftest()
    elif a.cmd == 'train': train(a)
    else:
        ck = torch.load(a.ckpt, map_location='cpu'); model = PixelRefiner(w=ck['args'].get('width', 32)); model.load_state_dict(ck['model']); model.to(a.device)
        r = evaluate(model, a.dump, meta_by_sid(), torch.device(a.device), max_clips=a.max_clips, save=a.save, save_points=a.save_points)
        print(json.dumps(r, indent=1))
