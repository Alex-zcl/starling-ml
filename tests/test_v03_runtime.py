"""Сквозные проверки публичного API, анализа, lifecycle и resume."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
from starling_ml import Engine, get_config, STANDARD_CONFIGS, analyze_config, save_config, read_config
from starling_ml.checkpoint import save_checkpoint, load_checkpoint
from starling_ml.core.context import Context
from starling_ml.ops.pointwise import categorical_loss_objective


class RuntimeTests(unittest.TestCase):
    def test_all_standard_recipes(self):
        for task in STANDARD_CONFIGS:
            with self.subTest(task=task):
                engine=Engine(get_config(task,max_steps=2)).run()
                self.assertEqual(engine.context.data['run.step'],2)
                self.assertTrue(torch.isfinite(engine.context.data['train.loss']))

    def test_configs_are_independent(self):
        first=get_config();first['modules']['optimizer']['params']['lr']=12
        self.assertEqual(get_config()['modules']['optimizer']['params']['lr'],.001)

    def test_analyzer_preserves_config(self):
        cfg=get_config();before=deepcopy(cfg)
        self.assertTrue(analyze_config(cfg).ok);self.assertEqual(cfg,before)

    def test_analyzer_unknown_setup_parameter(self):
        cfg=get_config();cfg['modules']['loss']['params']['typo']=1
        report=analyze_config(cfg)
        self.assertIn('SETUP_SIGNATURE',[i.code for i in report.errors])

    def test_analyzer_unknown_top_key(self):
        cfg=get_config();cfg['pipline']=[]
        self.assertFalse(analyze_config(cfg).ok)

    def test_analyzer_cycle(self):
        cfg=get_config()
        cfg['modules']['model']['params']['args']=['$ctx:model.instance']
        cfg['contracts']['model']['reads']=['model.instance']
        self.assertIn('SETUP_CYCLE',[i.code for i in analyze_config(cfg).errors])

    def test_analyzer_ambiguous_weights(self):
        cfg=get_config();cfg['modules']['dice']['params'].update(empty_target='ignore',empty_cost=.1,element_weight=2)
        codes={i.code for i in analyze_config(cfg).warnings}
        self.assertTrue({'EMPTY_IGNORED','OVERLAP_CANCELLATION'}<=codes)

    def test_analyzer_metadata(self):
        report=analyze_config(get_config(),metadata=dict(class_names=['a','b'],task_mode='multiclass',contains_background=True,split='validation',sampling_changed=True))
        codes={i.code for i in report.warnings}
        self.assertTrue({'NO_BACKGROUND','STATISTICS_SPLIT','SAMPLING_DISTRIBUTION'}<=codes)

    def test_analyzer_invalid_output(self):
        report=analyze_config(get_config('classification'),batch={'train.loss':torch.ones(2)})
        self.assertIn('SCALAR_LOSS',{i.code for i in report.errors})

    def test_frozen_nested_constants(self):
        ctx=Context({'nested':{'list':[1,2]}},{})
        with self.assertRaises(TypeError):ctx.constants['nested']['list'][0]=4

    def test_yaml_roundtrip(self):
        cfg=get_config('classification')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.yaml';save_config(cfg,path)
            self.assertEqual(cfg,read_config(path));Engine(path).run()

    def test_resume_matches_uninterrupted(self):
        whole=Engine(get_config('classification',max_steps=6,seed=91)).run()
        first=Engine(get_config('classification',max_steps=3,seed=91)).run()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';save_checkpoint(first,path)
            resumed=load_checkpoint(Engine(get_config('classification',max_steps=6,seed=91)),path).run()
        for a,b in zip(whole.context.data['model.instance'].parameters(),resumed.context.data['model.instance'].parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(resumed.context.data['run.step'],6)

    def test_exact_accumulation_matches_full_batch(self):
        cfg=get_config('classification',max_steps=1,seed=17)
        cfg['modules']['batch']['params']['batch_size']=8
        large=Engine(cfg).run()
        cfg['modules']['batch']['params']['batch_size']=4
        cfg['modules']['optimizer']['params']['accumulation']=2
        micro=Engine(cfg).run()
        for a,b in zip(large.context.data['model.instance'].parameters(),micro.context.data['model.instance'].parameters()):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)

    def test_checkpoint_rejects_pending_gradients(self):
        cfg=get_config('classification');cfg['modules']['optimizer']['params']['accumulation']=2
        engine=Engine(cfg).setup()
        for name in ('batch','move_input','move_target','forward','loss','optimizer'):engine.modules[name]()
        with self.assertRaises(RuntimeError):engine.state_dict()
        engine.close()

    def test_detached_loss_diagnostic(self):
        engine=Engine(get_config('classification')).setup()
        engine.context.view('loss')['train.loss']=torch.tensor(1.)
        with self.assertRaisesRegex(RuntimeError,'detached'):engine.modules['optimizer']()
        engine.close()

    def test_closed_engine_cannot_run_again(self):
        engine=Engine(get_config('classification',max_steps=0)).run()
        with self.assertRaises(RuntimeError):engine.run()

    def test_stop_prevents_validation_restart(self):
        engine=Engine(get_config('classification')).setup()
        engine.signal('stop_requested');engine.signal('validation_completed');engine.signal('step_completed')
        self.assertEqual(engine.context.data['run.phase'],'finished');self.assertFalse(engine.context.data['run.validation_due'])
        engine.close()

    def test_weight_product_yaml_order(self):
        cfg=dict(constants={},modules={
            'product':{'class':'starling_ml.modules.control.weights.WeightProduct','params':dict(inputs=['base','schedule'],output='effective')},
            'frequency':{'class':'starling_ml.modules.control.weights.FrequencyWeights','params':dict(frequency=[.2,.8],output='base')},
            'schedule':{'class':'starling_ml.modules.control.weights.LinearWeightSchedule','params':dict(output='schedule',start=1,target=2,steps=2)},
            'run':get_config()['modules']['run']},
            contracts={'product':dict(reads=['base','schedule'],creates=['effective']), 'frequency':dict(creates=['base']), 'schedule':dict(reads=['run.step'],creates=['schedule']), 'run':get_config()['contracts']['run']}, pipeline=[])
        cfg['modules']['run']['params']['max_steps']=2
        engine=Engine(cfg).setup();engine.signal('run_started');engine.signal('step_completed')
        torch.testing.assert_close(engine.context.data['effective'],engine.context.data['base']*1.5)
        engine.close()


if __name__=='__main__':unittest.main()
