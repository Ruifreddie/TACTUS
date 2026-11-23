#!/usr/bin/env python3
"""HF clips -> per-clip arrays, with UniTAP's own exporter (16 uniform frames, content crop, uid contact tracks, action range).

Sids: Gold 000000.., Silver 100000.. (stable: sorted by task, start, end). Writes into a shared root; repeated runs skip finished clips.
Crop: one fixed box per video from video_boxes.py, so a clip's crop never depends on its first frame or its labels.
"""
import argparse, collections, json, os, sys
from multiprocessing import Pool
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--unitap', required=True); ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--split-dir', required=True); ap.add_argument('--silver-index', required=True)
ap.add_argument('--groups', default='gold,silver', help='gold, silver or gold,silver'); ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--boxes', required=True, help='video_boxes.json')
a = ap.parse_args()
sys.path.insert(0, f'{a.unitap}/tools')
import export_case32 as EX  # noqa: E402
EX.RAW = Path(a.raw); EX.OUT = Path(a.out); EX.TOOLS_DIR = Path(a.unitap) / 'third_party/vitpose_dataset_tools'


def clip_key(e):
    return e['task'] + '/' + Path(e['anno_dir']).stem


def order(e):
    return (e['task'], e['start'], e['end'], e['anno_dir'])


def tasks():
    out = []
    gold = {s: json.load(open(f'{a.split_dir}/index_gold_{s}.json')) for s in ('train', 'val', 'test')}
    rows = sorted(((order(e), s, e) for s, d in gold.items() for e in d.values()), key=lambda r: r[0])
    sid = {o: f'{i:06d}' for i, (o, _, _) in enumerate(rows)}
    if 'gold' in a.groups:
        out += [(sid[o], clip_key(e), s) for o, s, e in rows]
    if 'silver' in a.groups:
        sil = sorted(json.load(open(a.silver_index)).values(), key=order)
        out += [(f'{100000 + i:06d}', clip_key(e), 'silver') for i, e in enumerate(sil)]
    return out


BOXES = json.load(open(a.boxes))


def one(t):
    box = tuple(BOXES[t[1].split('/')[0]]['box'])
    EX.builder.POINT_AWARE_CROP = False            # the box stays fixed; points outside it clip to its edge
    EX.builder.detect_content_bbox = lambda _f: box
    return EX.process_one(t)


def main():
    all_tasks = tasks(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); EX.make_dirs(out)
    meta_path = out / 'meta.json'
    done = {m['sid']: m for m in json.load(open(meta_path))} if meta_path.exists() else {}
    fails, stats = [], collections.Counter()
    # One line per finished clip, so a crash only costs the clips that were in progress.
    journal_path = out / 'meta_journal.jsonl'
    if journal_path.exists():
        for line in open(journal_path):
            try: j = json.loads(line)
            except ValueError: continue     # a line cut short by the crash
            done[j['row']['sid']] = j['row']; stats.update(j['stats'])
    journal = open(journal_path, 'a')
    todo = [t for t in all_tasks if t[0] not in done]
    print(f'export: {len(all_tasks)} clips, {len(todo)} to do', flush=True)
    with Pool(a.workers, initializer=EX._init_worker) as pool:
        for n, r in enumerate(pool.imap_unordered(one, todo, chunksize=4), 1):
            stats.update(r['stats'])
            if r['ok']:
                done[r['row']['sid']] = r['row']; journal.write(json.dumps({'row': r['row'], 'stats': r['stats']}) + '\n'); journal.flush()
            else: fails.append({k: r[k] for k in ('sid', 'clip_key', 'error')})
            if n % 500 == 0: print(f'[{n}/{len(todo)}] fail={len(fails)}', flush=True)
    rows = sorted(done.values(), key=lambda m: m['sid'])
    tmp = str(meta_path) + '.tmp'; json.dump(rows, open(tmp, 'w')); os.replace(tmp, meta_path)
    EX._init_worker(); b = EX.builder
    json.dump({'tool2id': b.TOOL2ID, 'act2id': b.ACT2ID, 'combo2id_6class': b.COMBO2ID, 'tool2topology': b.TOOL2TOPO, 'target_frames': 16,
               'contact_slots': 4, 'visibility': 'hard_binary'}, open(out / 'labels.json', 'w'), indent=1)
    for s in ('train', 'val', 'test', 'silver'):
        (out / f'all_{s}.txt').write_text(''.join(r['sid'] + '\n' for r in rows if r['split'] == s))
    (out / 'all.txt').write_text(''.join(r['sid'] + '\n' for r in rows))
    json.dump({'stats': dict(stats), 'n_ok': len(rows), 'failures': fails}, open(out / f"export_report_{a.groups.replace(',', '_')}.json", 'w'), indent=1)
    print(f'done ok={len(rows)} fail={len(fails)} stats={dict(stats)}', flush=True)
    for f in fails[:5]: print(f['clip_key'], f['error'][-400:])


if __name__ == '__main__':
    main()
