"""A small supervised policy: 77 inputs -> 32 hidden units -> 2 direction scores."""
import torch
from torch import nn

from dataset import ACTIONS, INPUT_DIM, encode_history, legal_actions


class DirectionModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.register_buffer('mean', torch.zeros(INPUT_DIM))
        self.register_buffer('scale', torch.ones(INPUT_DIM))
        self.layers = nn.Sequential(nn.Linear(INPUT_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, x, legal):
        scores = self.layers((x - self.mean) / self.scale)
        return scores.masked_fill(~legal, -1e9)


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = DirectionModel(checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


@torch.inference_mode()
def predict_action(model, observations, actions):
    features = torch.tensor([encode_history(observations, actions)], dtype=torch.float32)
    angle = round(observations[-1]['relative_actual_deg'] / 5) * 5
    legal = torch.tensor([legal_actions(angle)], dtype=torch.bool)
    scores = model(features, legal)[0]
    return ACTIONS[int(scores.argmax())]
