"""Create final-layer V-JEPA2.1 token caches without deleting input frames."""

import argparse
from pathlib import Path
import numpy as np
import torch
from affordance import encode_video
from affordance.vjepa_wrapper import VJEPAWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vjepa-repo', required=True)
    parser.add_argument('--frames-root', type=Path, required=True)
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--color-order', choices=['rgb', 'bgr'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    args = parser.parse_args()
    encoder = VJEPAWrapper(vjepa_repo=args.vjepa_repo, arch='vjepa2_1_vit_large_384',
                           img_size=384, num_frames=16, patch_size=16, tubelet_size=2,
                           hf_repo='', pretrained_from_hub=True, freeze=True).to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sid in args.split_file.read_text().splitlines():
        sid = sid.strip()
        if not sid:
            continue
        if Path(sid).name != sid or sid in ('.', '..'):
            raise ValueError('Sample IDs must be simple filenames.')
        frames = np.load(args.frames_root / f'{sid}.npy', allow_pickle=False)
        if args.color_order == 'bgr':
            frames = frames[..., ::-1].copy()
        tokens = encode_video(encoder, torch.from_numpy(frames)[None])[0].cpu().numpy()
        with (args.output_dir / f'{sid}.npy').open('xb') as stream:
            np.save(stream, tokens.astype(args.dtype))
        print(f'Cached {sid}', flush=True)


if __name__ == '__main__':
    main()
