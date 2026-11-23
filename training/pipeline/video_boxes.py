#!/usr/bin/env python3
"""One crop box per video, from what changes over time. Black bars and burned-in panels stay still; the surgical view does not.

Takes the middle frame of clips spread over the video, builds a per-pixel temporal std map, and trims each side down to the first row or
column whose motion (60th percentile) reaches a share of the motion at the frame's centre. The percentile keeps the top and bottom of a
circular endoscope view, where most of the row is black. Labels are never read, so train, val, test and Silver get the same rule.
usage: video_boxes.py --raw RAW --out boxes.json [--clips-per-video 40]"""
import argparse, glob, json, os
from multiprocessing import Pool
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True); ap.add_argument('--clips-per-video', type=int, default=40); ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--ratio', type=float, default=0.35); ap.add_argument('--q', type=float, default=60)
a = ap.parse_args()


def side(score, thr):
    """First and last index where a 9-long running median of the score passes."""
    k = 9; p = np.pad(score, k // 2, mode='edge'); med = np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=1); ok = np.where(med >= thr)[0]
    return (int(ok[0]), int(ok[-1]) + 1) if len(ok) else (0, len(score))


def one(task):
    cv2.setNumThreads(1); fs = sorted(glob.glob(f'{a.raw}/{task}/clips/*.mp4')); step = max(1, len(fs) / a.clips_per_video); ims = []; used = 0
    for i in np.arange(0, len(fs), step):
        cap = cv2.VideoCapture(fs[int(i)]); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2)); ok, f = cap.read(); cap.release()
        if ok: ims.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)); used += 1
    H, W = ims[0].shape; S = np.stack(ims).std(0); thr = max(6.0, a.ratio * float(np.median(S[H // 3:2 * H // 3, W // 3:2 * W // 3])))
    x0, x1 = side(np.percentile(S, a.q, axis=0), thr)        # columns first: a side panel must not dilute the row test
    y0, y1 = side(np.percentile(S[:, x0:x1], a.q, axis=1), thr)
    x0, x1 = side(np.percentile(S[y0:y1], a.q, axis=0), thr)
    flag = '' if (x1 - x0) >= .5 * W and (y1 - y0) >= .5 * H and used >= 8 else 'CHECK'
    return task, dict(box=[x0, y0, x1, y1], frame=[W, H], frames_used=used, flag=flag)


tasks = sorted(d for d in os.listdir(a.raw) if os.path.isdir(f'{a.raw}/{d}/clips'))
with Pool(a.workers) as p: out = dict(p.map(one, tasks))
# SurgVU comes from one recording system, so its side bars and bottom panel sit in the same place in every video; a quiet video should not keep them.
case = [v for k, v in out.items() if k.startswith('case_')]
if case:
    x0, x1 = int(np.median([v['box'][0] for v in case])), int(np.median([v['box'][2] for v in case]))
    low = [v['box'][3] for v in case if v['box'][3] < .96 * v['frame'][1]]; y1 = int(np.median(low)) if low else None
    for v in case: v['box'] = [x0, v['box'][1], x1, min(v['box'][3], y1) if y1 else v['box'][3]]
json.dump(out, open(a.out, 'w'), indent=1); print('boxes', len(out), 'flagged', [k for k, v in out.items() if v['flag']])
