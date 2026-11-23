#!/usr/bin/env python3
"""Token files <-> tar shards in persistent storage (GV). A shard per 500 files, so object storage sees few large files and a re-run only adds new ones.
pack SRC_DIR DST_DIR PREFIX | unpack DST_DIR SRC_DIR PREFIX"""
import glob, os, shutil, subprocess, sys, tempfile, time
op, a, b, pre = sys.argv[1:5]
if op == 'pack':
    os.makedirs(b, exist_ok=True); fs = sorted(os.path.basename(f) for f in glob.glob(f'{a}/*.npy') if not os.path.islink(f)); N = 500
    for i in range(0, len(fs), N):
        part = fs[i:i + N]; name = f'{b}/{pre}_{part[0][:-4]}_{part[-1][:-4]}_{len(part)}.tar'
        if os.path.exists(name): continue
        tmp = f'{os.path.dirname(a.rstrip("/"))}/.{pre}_{i}.tar'
        subprocess.run(['tar', '-cf', tmp, '-C', a, '-T', '/dev/stdin'], input='\n'.join(part).encode(), check=True)
        for k in range(6):  # a network volume's client can fail under load
            try:
                shutil.copyfile(tmp, name)
                if os.path.getsize(name) == os.path.getsize(tmp): break
                raise OSError('size differs')
            except OSError as ex:
                if k == 5: raise
                print('retry', name, repr(ex)[:120], flush=True); time.sleep(20 * (k + 1))
        os.remove(tmp)
    print(pre, len(fs), 'files in', len(glob.glob(f'{b}/{pre}_*.tar')), 'shards')
else:
    os.makedirs(b, exist_ok=True); n = 0
    for t in sorted(glob.glob(f'{a}/{pre}_*.tar')):
        subprocess.run(['tar', '-xf', t, '-C', b, '--skip-old-files'], check=True); n += 1
    print('unpacked', n, 'shards to', b)
