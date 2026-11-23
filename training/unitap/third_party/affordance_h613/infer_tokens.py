"""Predict a region sequence from one cached feature grid."""

import argparse
import numpy as np
import torch
from affordance import build_model, predict_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokens', required=True)
    parser.add_argument('--tool-id', type=int, required=True)
    parser.add_argument('--action-id', type=int, required=True)
    parser.add_argument('--combo-id', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    tokens = torch.from_numpy(np.load(args.tokens, allow_pickle=False)).unsqueeze(0)
    model = build_model(args.checkpoint, args.device)
    quad = predict_regions(model, tokens, [args.tool_id], [args.action_id], [args.combo_id])
    with open(args.output, 'xb') as stream:
        np.save(stream, quad[0].cpu().numpy())


if __name__ == '__main__':
    main()
