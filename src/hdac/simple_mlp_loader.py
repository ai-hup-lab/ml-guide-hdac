"""Rebuilding a saved SimpleMLP.

The ``.pt`` files are a state_dict plus metadata, not a pickled module, so the
network has to be constructed before the weights are loaded. The shape is read
from the checkpoint rather than assumed, so a head trained at a different width
still loads.
"""
import os
import sys

import torch

# src/models/simple_mlp.py holds the architecture; keep one definition of it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.simple_mlp import SimpleMLP  # noqa: E402


def load_mlp(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device).eval()
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported MLP checkpoint type in {path}: {type(checkpoint)}')

    state = checkpoint.get('state_dict', checkpoint)
    if 'net.0.weight' not in state:
        raise ValueError(f'{path} does not contain SimpleMLP weights')

    model = SimpleMLP(
        input_dim=int(checkpoint.get('input_dim', state['net.0.weight'].shape[1])),
        hidden_dim=int(checkpoint.get('hidden_dim', state['net.0.weight'].shape[0])),
        output_dim=int(checkpoint.get('output_dim', 1)),
        # Inactive at inference, so the default only matters for reconstruction.
        dropout=float(checkpoint.get('dropout', 0.0)),
    )
    model.load_state_dict(state)
    return model.to(device).eval()
