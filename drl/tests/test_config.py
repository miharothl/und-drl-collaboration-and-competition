import json

import pytest

from drl.experiment.config.ddpg_config import DdpgConfig
from drl.experiment.config.environment_config import EnvironmentConfig
from drl.experiment.config.experiment_config import ExperimentConfig, AgentConfig
from drl.experiment.config.master_config import MasterConfig
from drl.experiment.config.neural_network_config import NeuralNetworkConfig
from drl.experiment.config.reinforcement_learning_config import ReinforcementLearningConfig
from drl.experiment.config.replay_memory_config import ReplayMemoryConfig
from drl.experiment.config.trainer_config import TrainerConfig


class TestConfig:

    def test_setCurrent_setsConfig_getConfigReturnsCorrectConfig(self):
        agent_cfg = AgentConfig(
            action_size=4,
            state_size=8,
            discrete=True,
            state_rgb=False,
            num_frames=1,
        )

        environment_cfg = EnvironmentConfig(env_type='gym', num_agents=1)

        trainer_cfg = TrainerConfig(
            max_steps=1000000,
            max_episode_steps=1000,
            eval_frequency=20000,
            eval_steps=3000,
            human_flag=False,
            batch_size=64,
            update_every=4,
            num_updates=1,
            tau=0.001,
            gamma=0.99
        )

        neural_network_cfg = NeuralNetworkConfig(hidden_layers=[64, 64])

        actor_model_cfg = NeuralNetworkConfig(hidden_layers=[64, 64])
        critic_network_cfg = NeuralNetworkConfig(hidden_layers=[128, 128])

        ddpg_cfg = DdpgConfig(
            epsilon_start=1,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            lr_actor=1e-4,
            lr_critic=3e-4,
            weight_decay=0.0001,
            actor_model_cfg=actor_model_cfg,
            critic_model_cfg=critic_network_cfg
        )

        reinforcement_learning_cfg = ReinforcementLearningConfig(
            algorithm_type='ddpg',
            dqn_cfg=None,
            ddpg_cfg=ddpg_cfg)

        replay_memory_cfg = ReplayMemoryConfig(
            buffer_size=100000,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta0=0.4,
            prioritized_replay_eps=1e-6
        )

        experiment_cfg = ExperimentConfig(
            id='exp_id_1',
            gym_id='gym_exp_id_1',
            agent_cfg=agent_cfg,
            environment_cfg=environment_cfg,
            trainer_cfg=trainer_cfg,
            reinforcement_learning_cfg=reinforcement_learning_cfg,
            replay_memory_cfg=replay_memory_cfg
        )

        experiment_cfg_2 = ExperimentConfig(
            id='exp_id_2',
            gym_id='gym_exp_id_2',
            agent_cfg=agent_cfg,
            environment_cfg=environment_cfg,
            trainer_cfg=trainer_cfg,
            reinforcement_learning_cfg=reinforcement_learning_cfg,
            replay_memory_cfg=replay_memory_cfg
        )

        master_cfg = MasterConfig(experiment_cfgs=[experiment_cfg, experiment_cfg_2])

        # Serializing
        data = json.dumps(master_cfg, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        print(data)
        # Deserializing
        decoded_cfg = MasterConfig.from_json(json.loads(data))
        print(decoded_cfg)
        print(decoded_cfg.experiment_cfgs)

        decoded_cfg.set_current('exp_id_1')
        current = decoded_cfg.get_current()

        assert current.agent_cfg == agent_cfg
        assert current.environment_cfg == environment_cfg
        assert current.trainer_cfg == trainer_cfg
        assert current.reinforcement_learning_cfg == reinforcement_learning_cfg
        assert current.replay_memory_cfg == replay_memory_cfg

    def test_fromJson_configForAlghoritmIsMissing_raisesException(self):
        cfg = \
            {
                "experiment_cfgs": [
                    {
                        "agent_cfg": {
                            "action_size": 4,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 8
                        },
                        "environment_cfg": {
                            "env_type": "gym",
                            "num_agents": 1
                        },
                        "id": "exp_id_1",
                        "gym_id": "gym_exp_id_1",
                        "neural_network_cfg": {
                            "hidden_layers": [
                                64,
                                64
                            ]
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "ddpg",
                            "ddpg_cfg": None
                            #     {
                            #     "lr_actor": 0.0001,
                            #     "lr_critic": 0.0003,
                            #     "weight_decay": 0.0001
                            # }
                        },
                        "replay_memory_cfg": {
                            "buffer_size": 100000,
                            "prioritized_replay": True,
                            "prioritized_replay_alpha": 0.6,
                            "prioritized_replay_beta0": 0.4,
                            "prioritized_replay_eps": 1e-06
                        },
                        "trainer_cfg": {
                            "batch_size": 64,
                            "eval_frequency": 20000,
                            "eval_steps": 3000,
                            "gamma": 0.99,
                            "human_flag": False,
                            "max_episode_steps": 1000,
                            "max_steps": 1000000,
                            "tau": 0.001,
                            "update_every": 4,
                            "num_updates": 1
                        }
                    },
                ]
            }

        with pytest.raises(Exception):
            decoded_cfg = MasterConfig.from_json(cfg)


    def test_fromJson_configForDqn_configLoaded(self):
        cfg = \
            {
                "experiment_cfgs": [
                    {
                        "agent_cfg": {
                            "action_size": 4,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 8
                        },
                        "environment_cfg": {
                            "env_type": "gym",
                            "num_agents": 1
                        },
                        "id": "exp_id_1",
                        "gym_id": "gym_exp_id_1",
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn",
                            "dqn_cfg": {
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.1,
                                "epsilon_decay": 0.995,
                                "lr": 0.00001,
                                "model_cfg": {
                                    "hidden_layers": [
                                        64,
                                        64
                                    ]
                                },
                            },
                            "ddpg_cfg": None
                            #     {
                            #     "lr_actor": 0.0001,
                            #     "lr_critic": 0.0003,
                            #     "weight_decay": 0.0001
                            # }
                        },
                        "replay_memory_cfg": {
                            "buffer_size": 100000,
                            "prioritized_replay": True,
                            "prioritized_replay_alpha": 0.6,
                            "prioritized_replay_beta0": 0.4,
                            "prioritized_replay_eps": 1e-06
                        },
                        "trainer_cfg": {
                            "batch_size": 64,
                            "eval_frequency": 20000,
                            "eval_steps": 3000,
                            "gamma": 0.99,
                            "human_flag": False,
                            "max_episode_steps": 1000,
                            "max_steps": 1000000,
                            "tau": 0.001,
                            "update_every": 4,
                            "num_updates": 1
                        }
                    },
                ]
            }

        decoded_cfg = MasterConfig.from_json(cfg)
        assert decoded_cfg.experiment_cfgs[0].id == 'exp_id_1'
