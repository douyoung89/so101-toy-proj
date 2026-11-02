"""Untrained API example: scan right first, stop when both values are confirmed."""
import math
from PIL import Image


def make_policy():
    direction = 1

    def policy(state):
        nonlocal direction
        latest = state['observations'][-1]
        # Replace this image loading example with model inference.
        with Image.open(latest['policy_image']) as frame:
            if frame.size != (224, 224):
                raise ValueError('Unexpected policy frame size')
        if all(latest['memory']['fields'][field]['status'] == 'confirmed' and
               latest['memory']['fields'][field]['value'] is not None
               for field in ['protein_g', 'sugar_g']):
            return 'STOP'
        relative = math.degrees(state['current_wrist_rad']-.75)
        if relative >= 39:
            direction = -1
        elif relative <= -69:
            direction = 1
        return 'RIGHT_5' if direction > 0 else 'LEFT_5'

    return policy
