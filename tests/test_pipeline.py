"""Small tests for information flow, teacher semantics and episode termination."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ACTIONS, ANGLES, INPUT_DIM, encode_history, memory_complete, read_json
from evaluate import ReplayEnvironment, rule_action, run_episode
from model import DirectionModel, predict_action
from scripts.prepare_data import teacher_target
from simulation.memory import InformationMemory


def readings(protein=None, sugar=None):
    return {name: {'value': value, 'confidence': 0.99 if value is not None else None,
                   'ambiguous': False, 'candidates': []}
            for name, value in zip(('protein_g', 'sugar_g'), (protein, sugar))}


def observation(angle=0):
    return {'relative_actual_deg': angle, 'wrist_deg': angle,
            'joint_state': {'qpos_rad': [0, 0, 0, 0, 0.75 + angle * 3.141592653589793 / 180, 0]},
            'memory': InformationMemory().observe('first', 0, readings())}


class PipelineTests(unittest.TestCase):
    def test_features_ignore_images_values_truth_and_future(self):
        before = observation()
        after = copy.deepcopy(before)
        after.update(image='unreadable/path.png', truth={'protein_g': 999}, future_ocr='LEAK')
        after['memory']['fields']['protein_g']['value'] = 999
        self.assertEqual(encode_history([before], []), encode_history([after], []))
        self.assertEqual(len(encode_history([before], [])), INPUT_DIM)

    def test_future_observation_is_not_accepted_as_history(self):
        with self.assertRaises(ValueError):
            encode_history([observation(), observation(5)], [])

    def test_missing_ocr_preserves_memory_and_conflict_needs_reconfirmation(self):
        memory = InformationMemory()
        memory.observe('a', 0, readings(20))
        self.assertEqual(memory.observe('b', 1, readings())['fields']['protein_g']['value'], 20)
        self.assertIsNone(memory.observe('c', 2, readings(21))['fields']['protein_g']['value'])
        self.assertIsNone(memory.observe('d', 3, readings(21))['fields']['protein_g']['value'])
        self.assertEqual(memory.observe('e', 4, readings(21))['fields']['protein_g']['value'], 21)

    def test_teacher_retains_wrong_memory_and_requires_three_moves_to_repair(self):
        memory = InformationMemory().observe('start', 0, readings(31))
        scene = {'truth': {'protein_g': 32, 'sugar_g': 2}, 'views': [
            {'angle': 0, 'readings': readings(32), 'invalid_reasons': []},
            {'angle': 5, 'readings': readings(32, 2), 'invalid_reasons': []}]}
        current = {'wrist_deg': 0, 'memory': memory}
        self.assertEqual(teacher_target(scene, current, 3)[0], [0.0, 1.0])
        self.assertEqual(teacher_target(scene, current, 2), (None, 'unsolvable'))

    def test_wrong_complete_memory_has_no_direction_target(self):
        memory = InformationMemory().observe('start', 0, readings(31, 2))
        scene = {'truth': {'protein_g': 32, 'sugar_g': 2}, 'views': []}
        self.assertEqual(teacher_target(scene, {'memory': memory}, 8), (None, 'complete_wrong'))

    def test_model_masks_impossible_boundary_action(self):
        import torch
        model = DirectionModel()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.layers[-1].bias[0] = 100
        self.assertEqual(predict_action(model, [observation(-70)], []), 'RIGHT_5')

    def test_complete_at_final_budget_step_is_stop(self):
        scene = {'scene_id': 'test', 'split': 'val', 'start_deg': 0, 'safe_angles': list(ANGLES),
                 'truth': {'protein_g': 32, 'sugar_g': 2}, 'views': [
                     {'angle': a, 'readings': readings(32 if a == 40 else None, 2)} for a in ANGLES]}
        episode = run_episode(scene, 'right_first', None, 'replay', 1)
        self.assertTrue(episode['success'])
        self.assertEqual(episode['moves'], 8)
        self.assertEqual(episode['observations'][-1]['action'], 'STOP')

    def test_budget_exhaustion_is_not_stop(self):
        scene = {'scene_id': 'test', 'split': 'val', 'start_deg': 0, 'safe_angles': list(ANGLES),
                 'truth': {'protein_g': 32, 'sugar_g': 2}, 'views': [
                     {'angle': a, 'readings': readings()} for a in ANGLES]}
        episode = run_episode(scene, 'right_first', None, 'replay', 1)
        self.assertFalse(episode['success'])
        self.assertEqual(episode['stop_reason'], 'budget_exhausted')
        self.assertIsNone(episode['observations'][-1]['action'])


if __name__ == '__main__':
    unittest.main()
