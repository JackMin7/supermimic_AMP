import os
import unittest
from unittest.mock import patch
from pathlib import Path
import mjlab.tasks
import src.tasks
from scripts.train import TrainConfig,run_train
from supermimic.cli import build_arguments
from supermimic.train_fast import arguments


class CompatibilityTest(unittest.TestCase):
    def test_wmrl_defaults_and_compatible_parameters_mapped(self):
        original=TrainConfig.from_task('Unitree-G1-AMP-Flat')
        cfg=TrainConfig.from_task('Unitree-G1-AMP-Flat','supermimic')
        self.assertEqual(original.backend,'wmrl')
        self.assertEqual(original.agent.algorithm.learning_rate,1e-7)
        self.assertEqual(original.agent.num_steps_per_env,2)
        cfg.env.scene.num_envs=64;cfg.agent.max_iterations=14000;cfg.agent.seed=9
        cfg.agent.algorithm.gamma=.97;cfg.agent.algorithm.lam=.9
        cfg.agent.num_steps_per_env=4
        cfg.supermimic.terminal_format='json'
        parsed,_=arguments(build_arguments('Unitree-G1-AMP-Flat',cfg,Path('/tmp/unused'), 'cuda:0'))
        self.assertEqual((parsed.num_envs,parsed.updates,parsed.seed,parsed.utd),(64,14000,9,.25))
        self.assertEqual((parsed.gamma,parsed.gae_lambda,parsed.logger,parsed.checkpoint_prefix),(.97,.9,'tensorboard','model'))
        self.assertEqual(parsed.terminal_format,'json')
        self.assertEqual(TrainConfig.from_task('Unitree-G1-AMP-Flat').agent.algorithm.learning_rate,1e-7)

    def test_default_and_legacy_alias_dispatch_to_wmrl(self):
        for backend in ('wmrl','supermimic'):
            cfg=TrainConfig.from_task('Unitree-G1-AMP-Flat',backend)
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','LOCAL_RANK':'0','RANK':'0'}), \
                 patch('supermimic.cli.run') as optimized:
                run_train('Unitree-G1-AMP-Flat',cfg,Path('/tmp/unused'))
                optimized.assert_called_once_with('Unitree-G1-AMP-Flat',cfg,Path('/tmp/unused'),'cuda:0')

    def test_removed_ppo_backend_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError,'PPO training backend has been removed'):
            TrainConfig.from_task('Unitree-G1-AMP-Flat','ppo')

    def test_unsupported_ppo_knob_is_not_silently_ignored(self):
        cfg=TrainConfig.from_task('Unitree-G1-AMP-Flat','supermimic')
        cfg.agent.algorithm.entropy_coef=.123
        with self.assertRaisesRegex(ValueError,'entropy_coef'):
            build_arguments('Unitree-G1-AMP-Flat',cfg,Path('/tmp/unused'),'cuda:0')


if __name__=='__main__':unittest.main()
