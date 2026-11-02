import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIELDS, complete, correct, policy_input, write, read
from ocr_memory import InformationMemory
from check_partial_scene import shortest_path
from collect_partial import choose


def readings(p=None, s=None):
    return {f: {'value': v, 'confidence': .99 if v is not None else None,
                'ambiguous': False, 'candidates': []} for f, v in zip(FIELDS, [p, s])}


def view(angle, p=None, s=None, bad=False):
    return {'command_relative_deg': angle, 'invalid_reasons': ['overlap'] if bad else [],
            'readings': readings(p, s)}


class CausalTests(unittest.TestCase):
    def test_nutrition_fail_is_not_complete(self):
        state = InformationMemory().observe('a', 0, readings(3))
        self.assertEqual(state['decision'], 'fail')
        self.assertFalse(complete(state))

    def test_missing_does_not_erase_and_conflicts_need_two_confirmations(self):
        mem = InformationMemory()
        mem.observe('a', 0, readings(20))
        self.assertEqual(mem.observe('b', 1, readings())['fields']['protein_g']['value'], 20)
        self.assertIsNone(mem.observe('c', 2, readings(21))['fields']['protein_g']['value'])
        self.assertIsNone(mem.observe('d', 3, readings(21))['fields']['protein_g']['value'])
        self.assertEqual(mem.observe('e', 4, readings(21))['fields']['protein_g']['value'], 21)

    def test_bfs_merges_two_views(self):
        path = shortest_path([view(0, s=2), view(5, p=32)], 0, {'protein_g':32, 'sugar_g':2})
        self.assertEqual(path, [0, 5])

    def test_bfs_cannot_cross_invalid_or_missing_angle(self):
        table = [view(0, s=2), view(5, bad=True), view(10, p=32)]
        self.assertIsNone(shortest_path(table, 0, {'protein_g':32, 'sugar_g':2}))

    def test_wrong_complete_stop_prevents_future_oracle_repair(self):
        table = [view(0, s=2), view(5, p=30), view(10, p=32)]
        self.assertIsNone(shortest_path(table, 0, {'protein_g':32, 'sugar_g':2}))

    def test_budget_is_moves_not_observations(self):
        table = [view(0, s=2), view(5), view(10, p=32)]
        self.assertIsNone(shortest_path(table, 0, {'protein_g':32, 'sugar_g':2}, budget=1))
        self.assertEqual(shortest_path(table, 0, {'protein_g':32, 'sugar_g':2}, budget=2), [0,5,10])

    def test_direction_reverses_at_boundary_and_keeps_direction(self):
        import random
        safe = [-5, 0, 5]
        a = choose('left_first', -5, 1, safe, random.Random(1), history=[{'action':'LEFT_5'}])
        self.assertEqual(a, 'RIGHT_5')
        self.assertEqual(choose('left_first', 0, 2, safe, random.Random(1), history=[{'action':a}]), 'RIGHT_5')

    def test_input_excludes_privileged_and_future_data(self):
        mem = InformationMemory().observe('a', 0, readings(s=2))
        obs = {'image':'views/a.png', 'policy_image':'views/a_224.png',
               'joint_state': {'qpos_rad':[0,0,0,0,.75,0]}, 'relative_actual_deg':0,
               'ocr':[], 'readings':readings(s=2), 'memory':mem,
               'truth': {'protein_g':32}, 'settings': {'hidden':1}, 'future_ocr':'LEAK'}
        state = policy_input([obs], [])
        self.assertEqual(state['remaining_moves'], 8)
        self.assertNotIn('truth', state['observations'][0])
        self.assertNotIn('future_ocr', state['observations'][0])
        self.assertNotIn('settings', state['observations'][0])


class PhysicsTests(unittest.TestCase):
    def test_seeded_settings_and_clipped_coupling_are_safe(self):
        from simulator import sample_settings, create_scene, Simulator, labels, baseline
        self.assertEqual(sample_settings(11), sample_settings(11))
        base = baseline()
        self.assertAlmostEqual(base['slope_m_per_rad'], -.024)
        self.assertAlmostEqual(base['half_size_m'][1]*2, .0096)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            settings, label = sample_settings(20261029), labels()[0]
            create_scene(directory, label, settings)
            sim = Simulator(directory, settings, label, render=False)
            try:
                for angle in [0, -70, 40, 0]:
                    result = sim.move(angle)
                    self.assertEqual(result['invalid_reasons'], [])
                    self.assertAlmostEqual(result['actual_relative_deg'], angle, delta=.01)
                    center = float(sim.data.qpos[sim.slide])+base['body_position_m'][1]
                    self.assertGreaterEqual(center, settings['center_limits_y_m'][0]-.0001)
                    self.assertLessEqual(center, settings['center_limits_y_m'][1]+.0001)
            finally:
                sim.close()


class EpisodeIntegrationTests(unittest.TestCase):
    def test_live_loop_memory_stop_export_and_image_tamper_audit(self):
        from unittest.mock import patch
        from PIL import Image
        from common import sha
        from collect_partial import run_episode
        from audit_partial import audit_episode

        class FakeSimulator:
            def __init__(self, *args, **kwargs):
                self.angle = 0
                self.time = 0

            def state(self):
                return {'qpos_rad':[0,0,0,0,.75+self.angle*3.141592653589793/180,0],
                        'qvel_rad_s':[0]*6, 'actuator_targets_rad':[0,0,0,0,.75+self.angle*3.141592653589793/180,0],
                        'simulation_time_s':self.time}

            def move(self, angle):
                before = self.state()
                self.angle, self.time = angle, self.time+2
                return {'command_relative_deg':angle, 'command_wrist_rad':self.state()['qpos_rad'][4],
                        'actual_relative_deg':angle, 'before':before, 'after':self.state(), 'invalid_reasons':[]}

            def capture(self, directory, stem, thresholds):
                from ocr_memory import parse, accepted
                directory.mkdir(parents=True, exist_ok=True)
                for name, size in [(stem, (960,720)), (stem+'_224', (224,224))]:
                    Image.new('RGB', size).save(directory/(name+'.png'))
                raw = [{'text':'SUGAR 2 g' if self.angle == 0 else 'PROTEIN 32 g',
                        'confidence':.99, 'bbox':[[0,0],[100,0],[100,20],[0,20]]}]
                return {'image':stem+'.png', 'policy_image':stem+'_224.png',
                        'image_sha256':sha(directory/(stem+'.png')), 'policy_image_sha256':sha(directory/(stem+'_224.png')),
                        'joint_state':self.state(), 'relative_actual_deg':self.angle,
                        'ocr':raw, 'readings':accepted(parse(raw), thresholds)}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp)
            directory = dataset/'scenes/scene_000001'
            config = {'max_moves':8, 'thresholds':{'protein_g':.7,'sugar_g':.5},
                      'ocr_size':[960,720], 'policy_size':[224,224]}
            write(dataset/'dataset_config.json', config)
            write(directory/'scene.json', {'scene_id':directory.name, 'safe_angles_deg':[-5,0,5]})
            write(directory/'truth.json', {'protein_g':32, 'sugar_g':2, 'settings':{}, 'label':{}})
            plan = {'episode_id':'ep', 'strategy':'right_first', 'seed':1, 'start_deg':0}
            with patch('simulator.Simulator', FakeSimulator):
                episode = run_episode(directory, config, plan)
            self.assertTrue(episode['success'])
            self.assertEqual(episode['moves'], 1)
            self.assertEqual(episode['terminal_action']['action'], 'STOP')
            self.assertEqual(audit_episode(directory, episode, config), [])
            state = policy_input(episode['observations'][:1], [])
            self.assertEqual(len(state['observations']), 1)
            self.assertIsNone(state['observations'][0]['memory']['fields']['protein_g']['value'])
            Image.new('RGB',(960,720),'red').save(directory/episode['observations'][0]['image'])
            self.assertIn('image_hash', audit_episode(directory, episode, config))


if __name__ == '__main__':
    unittest.main()
