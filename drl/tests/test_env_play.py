import pytest

from drl.experiment.configuration import Configuration
from drl.experiment.experiment import Experiment
from drl.tests.test_env_config import TestEnvConfig


class TestEnvPlay:

    def test_listEnvs_configExist_returnsEnvs(self):

        config = TestEnvConfig.get()
        experiment = Experiment(config)

        envs = experiment.list_envs()

        assert len(envs) > 1

    @pytest.mark.depends(name='test_play')
    def test_playDummy_configExist_playsWithDummyAgent(self):
        config = TestEnvConfig.get()
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            if config.get_current_exp_cfg().environment_cfg.env_type == 'unity':
                break

            # experiment.play_dummy(mode='rgb-array', model=None, num_episodes=3, num_steps=10)
