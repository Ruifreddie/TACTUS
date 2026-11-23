#!/usr/bin/env python3
"""Crop check on an export: one clip per video, one frame each. Left: the source frame with its box. Right: the frame V-JEPA receives.
Writes sheet_XX.jpg and crop_report.json. usage: crop_sheet.py --raw --export --boxes --out"""
import argparse, json, os
import cv2, numpy as np
ap = argparse.ArgumentParser()
for k in ('raw', 'export', 'boxes', 'out'): ap.add_argument('--' + k, required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
boxes = json.load(open(a.boxes)); meta = {}
for m in sorted(json.load(open(f'{a.export}/meta.json')), key=lambda m: m['sid']): meta.setdefault(m['case'], m)
rows = []; tiles = []; TW, TH = 300, 170
for case, m in sorted(meta.items()):
    cap = cv2.VideoCapture(f"{a.raw}/{case}/clips/{os.path.basename(m['ann_relpath'])[:-5]}.mp4"); ok, f = cap.read(); cap.release()
    if not ok: rows.append(dict(task=case, error='frame not readable')); continue
    x0, y0, x1, y1 = [int(v) for v in m['crop_bbox_xyxy']]; cx0, cy0, cx1, cy1 = boxes[case]['box']; H, W = f.shape[:2]
    v = np.asarray(np.load(f"{a.export}/video_16_npy/{m['sid']}.npy", mmap_mode='r')[0])
    g = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY); dark = lambda r: float((r <= 10).mean())
    border = dict(left=dark(g[:, :8]), right=dark(g[:, -8:]), top=dark(g[:8]), bottom=dark(g[-8:]))
    grew = (x0, y0, x1, y1) != (cx0, cy0, cx1, cy1)
    pts = json.load(open(f"{a.raw}/{m['ann_relpath']}"))['points']; vis = [p for p in pts if p.get('vis', True)]
    outside = [p for p in vis if not (x0 <= p['x'] < x1 and y0 <= p['y'] < y1)]
    rows.append(dict(points_outside=round(len(outside) / max(1, len(vis)), 4), task=case, split=m['split'], frame=[W, H], crop=[x0, y0, x1, y1], content=[cx0, cy0, cx1, cy1], grew=grew, keep_w=round((x1 - x0) / W, 3), keep_h=round((y1 - y0) / H, 3), dark_border={k: round(b, 2) for k, b in border.items()}))
    l = f.copy(); cv2.rectangle(l, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), max(2, W // 300)); lt = cv2.resize(l, (TW, TH))
    rt = cv2.resize(v, (TH, TH)); t = np.zeros((TH + 18, TW + TH + 4, 3), np.uint8); t[18:, :TW] = lt; t[18:, TW + 4:] = rt
    name = case.replace('youtube_cholecystectomy_chenyan_', 'yt').split('_video')[0]
    cv2.putText(t, f"{name} {m['split'][:2]} {W}x{H}" + (' GREW' if grew else ''), (2, 13), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 128, 255) if grew else (0, 255, 255), 1, cv2.LINE_AA); tiles.append(t)
json.dump(rows, open(f'{a.out}/crop_report.json', 'w'), indent=1)
C, R = 4, 7
for p in range(0, len(tiles), C * R):
    ts = tiles[p:p + C * R]; ts += [np.zeros_like(tiles[0])] * (C * R - len(ts))
    cv2.imwrite(f'{a.out}/sheet_{p // (C * R):02d}.jpg', np.vstack([np.hstack(ts[i:i + C]) for i in range(0, C * R, C)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('videos', len(rows), 'unreadable', sum('error' in r for r in rows), 'clips with annotated points outside the box', sum(r.get('points_outside', 0) > 0 for r in rows))
