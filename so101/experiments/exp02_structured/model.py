"""Two Linear layers; causal structured features only, no image encoder."""
import torch
from torch import nn

from core import features, feature_names, legal_mask, DIRECTIONS


class DirectionMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        dimension = len(feature_names())
        self.register_buffer('mean', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x, legal=None):
        logits = self.net((x-self.mean)/self.scale)
        return logits.masked_fill(~legal, -1e9) if legal is not None else logits


def load_model(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['feature_names'] != feature_names():
        raise ValueError('Checkpoint feature schema differs')
    model = DirectionMLP(payload['hidden'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model


class DirectionActor:
    def __init__(self, checkpoint):
        self.model = load_model(checkpoint)

    @torch.inference_mode()
    def direction(self, state):
        x = features(state)
        legal = legal_mask(state['visited_angles_deg'][-1])
        logits = self.model(torch.tensor([x]), torch.tensor([legal], dtype=torch.bool))[0]
        index = int(logits.argmax())
        return DIRECTIONS[index], {'x': x, 'masked_logits': logits.tolist(), 'legal': legal}
