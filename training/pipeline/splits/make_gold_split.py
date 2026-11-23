#!/usr/bin/env python3
"""Gold train/val/test split for TACTUS: 80/10/10 by annotated frames.

Whole videos go to one split; YouTube videos of one channel stay together.
Annotated frames = union of a video's clip windows (the count the paper reports).
Usage: make_gold_split.py --data DIR_WITH_INDEX_FILES --out OUT_DIR
Needs index_gold.json, index_silver.json, videos.csv. Same input gives the same split.
"""
import argparse, collections as C, csv, itertools, json, math, os, random

SEED = 20261101
TEST, VAL = 0.10, 0.10
MIN_VIDEOS = 3
TEST_N = {"case": [3], "cholec80": [3], "comprehensive": [1], "heichole": [1], "youtube_chenyan": [1, 2, 3]}
VAL_N = {"case": [3], "cholec80": [3], "comprehensive": [1], "heichole": [1], "youtube_chenyan": [1, 2, 3]}
# YouTube videos grouped by source channel (read from the frames). Videos of one channel never split.
# The labels keep the alphabetical order of the channel names, so the search order is unchanged.
YT = "youtube_cholecystectomy_chenyan_"
YT_CHANNELS = {"channel_b": ["00000", "00001", "00008"],
               "channel_d": ["00003", "00004", "00005", "00006", "00007", "00010", "00011", "00017"],
               "channel_a": ["00016", "00019"], "channel_c": ["00013", "00018"]}


def combo(e):
    return e["tool"] + "/" + e["action"]


def merged_frames(windows):
    total, last = 0, -1
    for s, e in sorted(windows):
        total += max(0, e - max(s, last + 1) + 1)
        last = max(last, e)
    return total


def load(data):
    gold = json.load(open(os.path.join(data, "index_gold.json")))
    seen, clips, dropped = set(), {}, []
    for u in sorted(gold):  # one entry per clip file: repeated paths hold one track file
        if gold[u]["clip_dir"] in seen:
            dropped.append(u)
        else:
            clips[u] = gold[u]
        seen.add(gold[u]["clip_dir"])
    rows = [r for r in csv.DictReader(open(os.path.join(data, "videos.csv"))) if r["split"] == "gold"]
    return clips, dropped, rows


def build_units(rows):
    chan = {YT + v: c for c, vs in YT_CHANNELS.items() for v in vs}
    units = C.defaultdict(lambda: C.defaultdict(list))
    for r in sorted(rows, key=lambda r: r["task"]):
        units[r["group"]][chan.get(r["task"], r["task"])].append(r["task"])
    return {g: list(u.values()) for g, u in units.items()}


class Scorer:
    def __init__(self, clips, rows):
        self.src = {r["task"]: r["group"] for r in rows}
        self.fps = {r["task"]: float(r["fps"]) for r in rows}
        w = C.defaultdict(list)
        self.by = C.defaultdict(list)
        for e in clips.values():
            n = e["end"] - e["start"] + 1
            w[e["task"]].append((e["start"], e["end"]))
            self.by[e["task"]].append((combo(e), n, n / self.fps[e["task"]]))
        self.merged = {t: merged_frames(v) for t, v in w.items()}

    def tally(self, tasks):
        t = C.Counter()
        for k in tasks:
            s = self.src[k]
            t["S", s] += self.merged[k]
            for c, n, sec in self.by[k]:
                t["C", c] += n
                t["SC", s, c] += n
                t["N", c] += 1
                t["L", c] += math.log(sec)
        return t

    def loss(self, tasks, pool, target):
        t, a, x = self.tally(tasks), pool, 0.0
        seen = C.Counter(c for c in {(k, b[0]) for k in tasks for b in self.by[k]})
        vids = C.Counter(c for _, c in seen)
        combos = {b[0] for v in self.by.values() for b in v}
        x += 100 * sum(max(0, MIN_VIDEOS - vids[c]) for c in combos)  # every combo in several videos
        for k in a:
            if k[0] == "N":
                continue
            if k[0] == "L":
                x += 20 * (t[k] / max(t["N", k[1]], 1) - a[k] / a["N", k[1]]) ** 2  # clip seconds
                continue
            r = (t[k] / a[k] - target) / target
            w = {"S": 4, "C": 4}.get(k[0]) or min(a[k], 30000) / 30000  # small cells weigh less
            x += w * r * r
        return x


def search(units, sc, counts, target, seed, iters=2000, restarts=30):
    rng = random.Random(seed)
    pool = sc.tally([t for us in units.values() for u in us for t in u])
    yt_units = units["youtube_chenyan"]
    yt = [c for m in range(1, len(yt_units) + 1) for c in itertools.combinations(yt_units, m)
          if sum(map(len, c)) in counts["youtube_chenyan"]]
    flat = lambda s: [t for u in s.values() for x in u for t in x]
    best, best_l = None, 1e9
    for _ in range(restarts):
        st = {g: rng.sample(units[g], counts[g][0]) for g in units if g != "youtube_chenyan"}
        st["youtube_chenyan"] = list(rng.choice(yt))
        cur = sc.loss(flat(st), pool, target)
        for i in range(iters):
            g = rng.choice(list(units))
            old = st[g]
            if g == "youtube_chenyan":
                st[g] = list(rng.choice(yt))
            else:
                j = rng.randrange(len(old))
                st[g] = old[:j] + old[j + 1:] + [rng.choice([u for u in units[g] if u not in old])]
            new = sc.loss(flat(st), pool, target)
            if new < cur or rng.random() < math.exp((cur - new) / max(0.05 * (1 - i / iters), 1e-4)):
                cur = new
                if cur < best_l:
                    best, best_l = {k: list(v) for k, v in st.items()}, cur
            else:
                st[g] = old
    return sorted(flat(best)), best_l


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    clips, dropped, rows = load(a.data)
    silver = json.load(open(os.path.join(a.data, "index_silver.json")))
    units = build_units(rows)
    sc = Scorer(clips, rows)
    test, l1 = search(units, sc, TEST_N, TEST, SEED)
    rest = {g: [u for u in us if not set(u) & set(test)] for g, us in units.items()}
    val, l2 = search(rest, sc, VAL_N, VAL / (1 - TEST), SEED + 1)
    tset, vset = set(test), set(val)
    train = sorted(r["task"] for r in rows if r["task"] not in tset | vset)
    assert not (tset & vset) and not (tset | vset) & {e["task"] for e in silver.values()}
    part = lambda s: {u: e for u, e in clips.items() if e["task"] in s}
    tr, va, te = part(set(train)), part(vset), part(tset)
    os.makedirs(a.out, exist_ok=True)
    dump = lambda n, o: json.dump(o, open(os.path.join(a.out, n), "w"), indent=1, ensure_ascii=False)
    dump("gold_split.json", {"seed": SEED, "basis": "merged annotated frames", "loss_test": l1, "loss_val": l2,
                             "train_videos": train, "val_videos": val, "test_videos": test,
                             "youtube_channels": YT_CHANNELS, "dropped_repeated_clip_uuids": dropped})
    dump("index_gold_train.json", tr)
    dump("index_gold_val.json", va)
    dump("index_gold_test.json", te)
    dump("index_gold_train_plus_silver.json", {**tr, **silver})
    n = C.Counter(e["task"] for e in clips.values())
    with open(os.path.join(a.out, "split_videos.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "source", "task", "clips", "annotated_frames", "fps", "width", "height"])
        name = lambda t: "test" if t in tset else "val" if t in vset else "train"
        for r in sorted(rows, key=lambda r: (name(r["task"]), r["group"], r["task"])):
            w.writerow([name(r["task"]), r["group"], r["task"], n[r["task"]], sc.merged[r["task"]],
                        r["fps"], r["width"], r["height"]])


if __name__ == "__main__":
    main()
