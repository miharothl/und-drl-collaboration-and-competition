import pytest

from drl.experiment.experiment import Experiment
from drl.tests.test_env_config import TestEnvConfig


class TestEnvTrain:

    @pytest.mark.depends(name='test_train')
    def test_train_environment_gymai_lunarlander(self):
        config = TestEnvConfig.get()
        experiment = Experiment(config)

        experiment.set_env('lunarlander')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_environment_atari_breakout(self):
        config = TestEnvConfig.get()
        experiment = Experiment(config)

        experiment.set_env('breakout')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_environment_atari_breakout_rgb(self):
        config = TestEnvConfig.get()
        experiment = Experiment(config)

        experiment.set_env('breakout-rgb')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_environment_unity_banana(self):
        config = TestEnvConfig.get()
        experiment = Experiment(config)

        experiment.set_env('banana')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)
