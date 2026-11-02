import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (FIELDS, DIRECTIONS, ANGLES, causal_state, features, feature_names,
                  direction_teacher, legal_mask, transition, compact)
from common import write
from ocr_memory import InformationMemory


def readings(p=None, s=None):
    return {f:{'value':v,'confidence':.99 if v is not None else None,
               'ambiguous':False,'candidates':[]} for f,v in zip(FIELDS,[p,s])}


def view(a,p=None,s=None,bad=False):
    return {'command_relative_deg':a,'readings':readings(p,s),'invalid_reasons':['bad'] if bad else []}


def memory(p=None,s=None):
    return InformationMemory().observe('actual_initial',0,readings(p,s))


def obs(angle=0,p=None,s=None):
    return {'joint_state':{'qpos_rad':[0,0,0,0,.75+math.radians(angle),0]},
            'relative_actual_deg':angle,'memory':memory(p,s)}


class FeatureTests(unittest.TestCase):
    def test_no_truth_future_image_strategy_or_current_target_leakage(self):
        observations = [obs()]
        baseline = features(causal_state(observations,[]))
        changed = copy.deepcopy(observations)
        changed[0].update(truth={'protein_g':32},future_ocr='LEAK',image='/must/not/open',
                          oracle_path=[0,5],strategy='shortest',target='LEFT_5',action='LEFT_5')
        changed[0]['memory']['fields']['protein_g']['value'] = 999
        self.assertEqual(baseline,features(causal_state(changed,[])))
        self.assertEqual(len(baseline),77)

    def test_past_action_is_input_and_history_is_padded(self):
        observations = [obs(0),obs(5)]
        left = features(causal_state(observations,[{'action':'LEFT_5'}]))
        right = features(causal_state(observations,[{'action':'RIGHT_5'}]))
        self.assertNotEqual(left,right)
        self.assertEqual(left[1],7/8)
        self.assertEqual(len(left),len(feature_names()))

    def test_future_extra_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            causal_state([obs(),obs(5)],[])

    def test_boundary_mask(self):
        self.assertEqual(legal_mask(-70),[False,True])
        self.assertEqual(legal_mask(40),[True,False])


class TeacherTests(unittest.TestCase):
    def test_starts_from_current_memory_instead_of_reset(self):
        result = direction_teacher([view(0),view(5,p=32)],0,memory(s=2),1,{'protein_g':32,'sugar_g':2})
        self.assertEqual(result['target'],[0.,1.])
        self.assertEqual(result['shortest_moves'],1)

    def test_tie_has_soft_targets(self):
        result = direction_teacher([view(-5,p=32),view(0),view(5,p=32)],0,memory(s=2),1,{'protein_g':32,'sugar_g':2})
        self.assertEqual(result['status'],'tie')
        self.assertEqual(result['target'],[.5,.5])

    def test_wrong_complete_stop_cannot_be_repaired_by_oracle(self):
        result = direction_teacher([view(0),view(5,p=31),view(10,p=32)],0,memory(s=2),8,{'protein_g':32,'sugar_g':2})
        self.assertEqual(result['status'],'unsolvable')

    def test_partial_misread_requires_conflict_and_two_new_confirmations(self):
        table=[view(0,p=32),view(5,p=32,s=2)]
        truth={'protein_g':32,'sugar_g':2}
        result=direction_teacher(table,0,memory(p=31),3,truth)
        self.assertTrue(result['partial_wrong'])
        self.assertEqual(result['shortest_moves'],3)
        self.assertEqual(direction_teacher(table,0,memory(p=31),2,truth)['status'],'unsolvable')

    def test_complete_wrong_and_budget_states_have_no_direction(self):
        truth={'protein_g':32,'sugar_g':2}
        self.assertEqual(direction_teacher([],0,memory(p=31,s=2),8,truth),{'status':'complete_wrong'})
        self.assertEqual(direction_teacher([],0,memory(),0,truth),{'status':'budget_exhausted'})

    def test_invalid_geometry_is_not_a_teacher_path(self):
        self.assertEqual(direction_teacher([view(5,p=32,bad=True)],0,memory(s=2),8,
                         {'protein_g':32,'sugar_g':2})['status'],'unsolvable')


class ProtocolTests(unittest.TestCase):
    def test_two_linear_layers_only(self):
        import torch
        from model import DirectionMLP
        model=DirectionMLP()
        self.assertEqual(sum(isinstance(m,torch.nn.Linear) for m in model.modules()),2)
        self.assertEqual(sum(p.numel() for p in model.parameters()),2562)

    def test_training_is_blocked_after_test_started(self):
        from train import train
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            write(run/'test_evaluation_started.json',{})
            with self.assertRaisesRegex(ValueError,'locked'):
                train(run)

    def test_complete_at_last_observation_is_stop_not_budget(self):
        from evaluate import live_episode
        import numpy as np

        class FakeSimulator:
            final_values=(32,2)

            def __init__(self,*args,**kwargs):
                self.reference=.75
                self.aids=[0]*6
                self.model=SimpleNamespace(joint=lambda name:SimpleNamespace(id=0),
                    jnt_range=np.array([[-math.pi,math.pi]]),actuator_ctrlrange=np.array([[-math.pi,math.pi]]))
                self.angle=0

            def state(self):
                return {'qpos_rad':[0,0,0,0,.75+math.radians(self.angle),0]}

            def move(self,a):
                before=self.state();self.angle=a
                return {'invalid_reasons':[],'before':before,'after':self.state()}

            def capture(self,directory,stem,thresholds):
                vals=self.final_values if self.angle==40 else (None,None)
                return {'image':stem+'.png','policy_image':stem+'_224.png','joint_state':self.state(),
                        'relative_actual_deg':self.angle,'readings':readings(*vals),'ocr':[]}

            def close(self):pass

        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);dataset=run/'pilot';scene=dataset/'scenes/s'
            write(dataset/'dataset_config.json',{'thresholds':{}})
            write(scene/'scene.json',{'safe_angles_deg':list(ANGLES),'label_id':'l'})
            write(scene/'truth.json',{'settings':{},'label':{},'protein_g':32,'sugar_g':2})
            checkpoint=run/'fake.pt';checkpoint.write_text('placeholder')
            case={'scene_id':'s','start_deg':0,'seed':1,'split':'test'}
            for values,reason,wrong in [((32,2),'memory_complete',False),
                                       ((None,None),'budget_exhausted',False),
                                       ((31,2),'memory_complete',True)]:
                FakeSimulator.final_values=values
                with patch('simulator.Simulator',FakeSimulator):
                    e=live_episode(dataset,run/'output',case,'right_first',checkpoint)
                self.assertEqual(e['moves'],8)
                self.assertEqual(e['stop_reason'],reason)
                self.assertEqual(e['wrong_stop'],wrong)
                self.assertEqual(e['budget_exhausted'],reason=='budget_exhausted')


class SavedRunAuditTests(unittest.TestCase):
    def test_saved_live_inputs_images_memory_and_actions(self):
        from core import HERE
        from common import read, sha
        from prepare import verify_preserved
        from ocr_memory import parse, accepted
        run=HERE/'runs/pilot_20261102'
        if not (run/'test_evaluation_complete.json').exists():
            self.skipTest('Run prepare/train/evaluate first for saved-run verification')
        config=read(Path(read(run/'experiment.json')['dataset'])/'dataset_config.json')
        paths=sorted((run/'evaluation').glob('*/*/*/episode.json'))
        self.assertEqual(len(paths),15)
        for path in paths:
            e=read(path);obs=e['observations'];actions=e['actions']
            self.assertEqual(len(obs),len(actions)+1)
            mem=InformationMemory()
            for i,o in enumerate(obs):
                parsed=accepted(parse(o['ocr']),config['thresholds'])
                self.assertEqual(parsed,o['readings'])
                self.assertEqual(mem.observe(o['observation_id'],i,parsed),o['memory'])
                for key in ['image','policy_image']:
                    frame=(path.parent/o[key]).resolve()
                    self.assertTrue(frame.is_relative_to(run))
                    self.assertEqual(sha(frame),o[key+'_sha256'])
                if i<len(actions):
                    a=actions[i]
                    expected=causal_state(obs[:i+1],actions[:i])
                    self.assertEqual(expected,a['decision_state'])
                    self.assertEqual(a['motion'],obs[i+1]['arrival'])
                    self.assertEqual(a['motion']['before'],o['joint_state'])
                    self.assertEqual(a['motion']['after'],obs[i+1]['joint_state'])
                    delta=-5 if a['action']=='LEFT_5' else 5
                    self.assertEqual(o['command_relative_deg']+delta,obs[i+1]['command_relative_deg'])
                    if e['method']=='mlp':
                        self.assertEqual(a['diagnostic']['x'],features(expected))
                        prediction=max(range(2),key=lambda j:a['diagnostic']['masked_logits'][j])
                        self.assertEqual(a['action'],DIRECTIONS[prediction])
            full=all(obs[-1]['memory']['fields'][f]['status']=='confirmed' and
                     obs[-1]['memory']['fields'][f]['value'] is not None for f in FIELDS)
            self.assertEqual(e['stop_executed'],full)
            if e['budget_exhausted']:
                self.assertEqual(len(actions),8)
                self.assertFalse(e['stop_executed'])
        self.assertEqual(verify_preserved(run)['modified_files'],[])


if __name__=='__main__':
    unittest.main()
