#!/usr/bin/env python3
"""A directory tree <-> tar shards of about 4 GB in persistent storage (GV), so the export of the CPU phase reaches the GPU machine.
Reads and writes retry, because a network volume's client can fail under load. A re-run only does what is missing, and unpack checks file count and bytes.
pack SRC_DIR VOL_DIR | unpack VOL_DIR SRC_DIR"""
import json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

op, a, b = sys.argv[1:4]; JOBS = int(os.environ.get('VOL_JOBS', 6)); SHARD = 4 << 30


def retry(fn, what):
    for i in range(6):
        try: return fn()
        except Exception as ex:
            print('retry', what, repr(ex)[:120], flush=True); time.sleep(20 * (i + 1))
    raise RuntimeError('gave up on ' + what)


def scan(root):
    files, dirs = [], []
    for d, ds, fs in os.walk(root):
        rel = os.path.relpath(d, root); ds.sort()
        if rel != '.': dirs.append(rel)
        files += [os.path.normpath(os.path.join(rel, f)) for f in sorted(fs)]
    return files, dirs


def total(root, files):
    return sum(os.lstat(os.path.join(root, f)).st_size for f in files)


if op == 'pack':
    os.makedirs(b, exist_ok=True); files, dirs = scan(a); shards, cur, size = [], [], 0
    for f in files:
        s = os.lstat(os.path.join(a, f)).st_size
        if cur and size + s > SHARD: shards.append(cur); cur, size = [], 0
        cur.append(f); size += s
    if cur: shards.append(cur)

    def pack1(t):
        i, part = t; name = f'{b}/export_{i:05d}.tar'
        if os.path.exists(name + '.ok'): return
        tmp = f'{os.path.dirname(a.rstrip("/"))}/.pack_{i}.tar'
        subprocess.run(['tar', '-cf', tmp, '-C', a, '-T', '-'], input=('\n'.join(part) + '\n').encode(), check=True)

        def up():
            shutil.copyfile(tmp, name)
            if os.path.getsize(name) != os.path.getsize(tmp): raise IOError('size differs')
        retry(up, name); open(name + '.ok', 'w').close(); os.remove(tmp)
        print('packed', i + 1, 'of', len(shards), flush=True)
    with ThreadPoolExecutor(JOBS) as ex: list(ex.map(pack1, enumerate(shards)))
    man = {'files': len(files), 'bytes': total(a, files), 'shards': len(shards), 'dirs': dirs}
    json.dump(man, open(f'{b}/MANIFEST.json', 'w'))
    print('packed', len(files), 'files,', man['bytes'] >> 30, 'GB in', len(shards), 'shards')
else:
    man = json.load(open(f'{a}/MANIFEST.json')); os.makedirs(b, exist_ok=True); mark = b.rstrip('/') + '.shards'; os.makedirs(mark, exist_ok=True)
    for d in man['dirs']: os.makedirs(os.path.join(b, d), exist_ok=True)

    def unpack1(i):
        m = f'{mark}/{i}'
        if os.path.exists(m): return
        retry(lambda: subprocess.run(['tar', '-xf', f'{a}/export_{i:05d}.tar', '-C', b], check=True), f'shard {i}')
        open(m, 'w').close(); print('unpacked', i + 1, 'of', man['shards'], flush=True)
    with ThreadPoolExecutor(JOBS) as ex: list(ex.map(unpack1, range(man['shards'])))
    files, _ = scan(b); got = total(b, files)
    if len(files) != man['files'] or got != man['bytes']: sys.exit(f"restored {len(files)} files, {got} bytes; expected {man['files']}, {man['bytes']}")
    print('restored', len(files), 'files,', got >> 30, 'GB')
