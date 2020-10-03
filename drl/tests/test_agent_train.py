import pytest

from drl.experiment.experiment import Experiment
from drl.tests.test_agent_config import TestAgentConfig


class TestEnvironmentTrain:


    def test_train_agent_dqn(self):
        config = TestAgentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('lunarlander-dqn')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_agent_dqn_with_frames(self):
        config = TestAgentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('lunarlander-dqn-withframes')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_agent_dqn_with_disabled_prioritized_replay(self):
        config = TestAgentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('lunarlander-dqn-noprio')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_agent_dqn_dueling(self):
        config = TestAgentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('lunarlander-dqn-dueling')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_agent_dqn_double(self):
        config = TestAgentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('lunarlander-dqn-double')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)

    def test_train_agent_ddpg(self):
        config = TestAgentConfig.get()
        experiment = Experiment(config)

        experiment.set_env('walker-ddpg')
        scores = experiment.train()
        assert len(scores) == (config.get_current_exp_cfg().trainer_cfg.max_steps / config.get_current_exp_cfg().trainer_cfg.max_episode_steps)
