"""Exercise data loading, backward, AdamW, EMA, save and eval on synthetic data."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def main():
    package = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='affordance_training_test_') as temporary:
        root = Path(temporary)
        data = root / 'data'
        (data / 'gt_ct_xy').mkdir(parents=True)
        (data / 'action_range_16').mkdir()
        rows = []
        rng = np.random.default_rng(613)
        for split in ('train', 'val'):
            sid = 'synthetic_' + split
            (data / f'{split}.txt').write_text(sid + '\n')
            cache = data / 'vjepa_token_cache' / split
            cache.mkdir(parents=True)
            np.save(cache / f'{sid}.npy', rng.normal(size=(8,24,24,1024)).astype('float16'))
            quad = np.tile(np.array([[.3,.3],[.6,.3],[.6,.6],[.3,.6]], dtype='float32'), (16,1,1))
            np.save(data / 'gt_ct_xy' / f'{sid}.npy', quad)
            np.savez(data / 'action_range_16' / f'{sid}.npz',
                     features=np.zeros((16,4), dtype='float32'),
                     inside_action_range=np.array([0]*4+[1]*8+[0]*4, dtype='float32'))
            rows.append(dict(sid=sid, case='synthetic_case_'+split, tool_id=0, act_id=0, combo_id=0))
        (data / 'meta.json').write_text(json.dumps(rows))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        base = [sys.executable, str(package/'train.py'), '--data-root', str(data),
                '--allow-custom-data', '--device', 'cpu', '--num-workers', '0', '--batch-size', '1']
        for mode, extra in [('train', ['--epochs', '1']),
                            ('eval', ['--checkpoint', str(root/'train/best.pt')])]:
            result = subprocess.run(base + ['--out-dir', str(root/mode)] + extra,
                                    env=env, capture_output=True, text=True)
            if result.returncode:
                print(result.stdout[-4000:])
                print(result.stderr[-4000:])
                raise SystemExit(result.returncode)
        metrics = json.loads((root/'train/best_metrics.json').read_text())
        assert np.isfinite(metrics['train_loss'])
        assert metrics['val_selection_source'] == 'ema'
        assert (root/'eval/eval_condition_ablation.json').is_file()
        print('PASS: synthetic train/backward/AdamW/EMA/checkpoint/evaluation cycle')


if __name__ == '__main__':
    main()
