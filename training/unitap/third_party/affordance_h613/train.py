"""Portable H613 training/evaluation launcher using the original training loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from affordance import training_core as core


def effective_args(options):
    values = json.loads(options.config.read_text())
    values.update(data_root=options.data_root, cache_root=options.cache_root or options.data_root / 'vjepa_token_cache',
                  out_dir=options.out_dir)
    for name in ['device', 'epochs', 'batch_size', 'num_workers', 'seed']:
        value = getattr(options, name)
        if value is not None:
            values[name] = value
    values['eval_only_checkpoint'] = options.checkpoint
    for name in ['data_root', 'cache_root', 'video_root', 'tool_mask_root', 'out_dir',
                 'eval_only_checkpoint', 'init_checkpoint', 'teacher_cache_root', 'write_teacher_cache_root']:
        if values.get(name) is not None:
            values[name] = Path(values[name])
    if values['model_mode'] != 'proposal_point_temporal':
        raise ValueError('This launcher is for the H613 affordance architecture.')
    if values.get('init_checkpoint') or values.get('teacher_ensemble_checkpoints') or values.get('teacher_cache_root'):
        raise ValueError('H613 replay starts the head from zero without a teacher.')
    if values.get('write_teacher_cache_root') or values.get('eval_ensemble_checkpoints'):
        raise ValueError('H613 replay does not use cache export or checkpoint ensembles.')
    return argparse.Namespace(**values)


def validate_data(args, custom=False):
    root = args.data_root
    rows = json.loads((root / 'meta.json').read_text())
    metadata = {str(row['sid']): row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError('Duplicate sample IDs in meta.json.')
    splits = {s: core.read_split(root, s) for s in ('train', 'val')}
    if set(splits['train']) & set(splits['val']):
        raise ValueError('Train/validation sample overlap.')
    case_sets = {}
    for split, ids in splits.items():
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f'{split}: empty split or duplicate IDs.')
        if not custom and len(ids) != {'train': 5622, 'val': 1471}[split]:
            raise ValueError('Historical counts differ; use --allow-custom-data for a different dataset.')
        cases = set()
        for sid in ids:
            if Path(sid).name != sid or sid in ('.', '..'):
                raise ValueError('Sample IDs must be simple filenames.')
            row = metadata[sid]
            cases.add(str(row['case']))
            for key, limit in [('tool_id', 9), ('act_id', 9), ('combo_id', 17)]:
                if not isinstance(row[key], int) or not 0 <= row[key] < limit:
                    raise ValueError(f'Invalid {key}.')
            token = np.load(args.cache_root / split / f'{sid}.npy', mmap_mode='r', allow_pickle=False)
            if token.shape != (8, 24, 24, 1024):
                raise ValueError(f'Invalid token shape for {sid}.')
            quad = np.load(root / 'gt_ct_xy' / f'{sid}.npy', allow_pickle=False)
            if quad.shape != (16, 4, 2) or not np.isfinite(quad).all() or (quad < 0).any() or (quad > 1).any():
                raise ValueError(f'Invalid normalized quad targets for {sid}.')
            if args.use_role_loss_weighting:
                with np.load(root / 'action_range_16' / f'{sid}.npz', allow_pickle=False) as role:
                    if role['features'].shape != (16, 4) or role['inside_action_range'].shape != (16,):
                        raise ValueError(f'Invalid action-range arrays for {sid}.')
        case_sets[split] = cases
    if case_sets['train'] & case_sets['val']:
        raise ValueError('Train/validation case overlap.')
    if not custom and (len(case_sets['train']), len(case_sets['val'])) != (46, 28):
        raise ValueError('Historical case counts differ.')
    return {s: len(v) for s, v in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/h613_train.json')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Evaluate a trusted local checkpoint instead of training.')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    for name in ['epochs', 'batch-size', 'num-workers', 'seed']:
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--allow-custom-data', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate data and print settings without training.')
    options = parser.parse_args()
    args = effective_args(options)
    counts = validate_data(args, options.allow_custom_data)
    if options.checkpoint is not None and not options.checkpoint.is_file():
        raise FileNotFoundError(options.checkpoint)
    if options.dry_run:
        print(json.dumps({'counts': counts, 'args': vars(args)}, indent=2, default=str))
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu explicitly for a small smoke test.')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Use a new output directory; existing runs are not overwritten.')
    core.parse_args = lambda: args
    core.main()


if __name__ == '__main__':
    main()
