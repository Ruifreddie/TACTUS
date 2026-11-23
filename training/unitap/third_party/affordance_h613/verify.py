"""Offline synthetic smoke test; optionally verify H613 checkpoint loading."""

import argparse
import json
import torch
from affordance import build_model, predict_regions, preprocess_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(613)
    model = build_model(args.checkpoint)
    tokens = torch.randn(2, 8, 24, 24, 1024)
    quad = predict_regions(model, tokens, [0, 1], [0, 1], [0, 1])
    assert quad.shape == (2, 16, 4, 2)
    assert torch.isfinite(quad).all() and (quad >= 0).all() and (quad <= 1).all()
    frames = torch.zeros(1, 16, 32, 48, 3, dtype=torch.uint8)
    normalized = preprocess_rgb(frames)
    assert normalized.shape == (1, 16, 3, 384, 384)
    torch.testing.assert_close(normalized[0, 0, :, 0, 0],
                               -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225]))
    for bad_tokens, bad_ids in [(tokens[:, :4], [0, 1]), (tokens, [9, 1]), (tokens, [0])]:
        try:
            predict_regions(model, bad_tokens, bad_ids, [0, 1], [0, 1])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input was accepted')
    print(json.dumps({'synthetic_forward': 'passed', 'shape': list(quad.shape),
                      'checkpoint_loaded': args.checkpoint is not None,
                      'preprocessing_and_validation': 'passed'}))


if __name__ == '__main__':
    main()
