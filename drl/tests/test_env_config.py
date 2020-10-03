import pytest

from drl.experiment.configuration import Configuration
from drl.experiment.experiment import Experiment


class TestEnvConfig:

    @staticmethod
    def get():
        cfg = \
            {
                "experiment_cfgs": [
                    {
                        "id": "lunarlander",
                        "gym_id": "LunarLander-v2",
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
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn",
                            "dqn_cfg": {
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.01,
                                "epsilon_decay": 0.995,
                                "lr": 0.0001,
                                "model_cfg": {
                                    "hidden_layers": [
                                        64,
                                        64
                                    ]
                                },
                            },
                            "ddpg_cfg": None
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
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4,
                            "num_updates": 1
                        }
                    },
                    {
                        "id": "breakout",
                        "gym_id": "Breakout-ram-v4",
                        "agent_cfg": {
                            "action_size": 3,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 128
                        },
                        "environment_cfg": {
                            "env_type": "spaceinvaders_atari_gym",
                            "num_agents": 1
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn",
                            "dqn_cfg": {
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.01,
                                "epsilon_decay": 0.995,
                                "lr": 0.0001,
                                "model_cfg": {
                                    "hidden_layers": [
                                        64,
                                        64
                                    ]
                                },
                            },
                            "ddpg_cfg": None
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
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4,
                            "num_updates": 1
                        }
                    },
                    {
                        "id": "breakout-rgb",
                        "gym_id": "Breakout-v4",
                        "agent_cfg": {
                            "action_size": 3,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": True,
                            "state_size": [80, 80]
                        },
                        "environment_cfg": {
                            "env_type": "spaceinvaders_atari_gym",
                            "num_agents": 1
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn",
                            "dqn_cfg": {
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.01,
                                "epsilon_decay": 0.995,
                                "lr": 0.0001,
                                "model_cfg": {
                                    "hidden_layers": [
                                        64,
                                        64
                                    ]
                                },
                            },
                            "ddpg_cfg": None
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
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4,
                            "num_updates": 1
                        }
                    },
                    {
                        'id': 'banana',
                        'gym_id': 'env/unity/mac/banana.app',
                        "agent_cfg": {
                            "action_size": 4,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 37
                        },
                        "environment_cfg": {
                            "env_type": "unity",
                            "num_agents": 1
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "dqn_double",
                            "dqn_cfg": {
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.01,
                                "epsilon_decay": 0.995,
                                "lr": 0.0001,
                                "model_cfg": {
                                    "hidden_layers": [
                                        64,
                                        64
                                    ]
                                },
                            },
                            "ddpg_cfg": None
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
                            "eval_frequency": 16,
                            "eval_steps": 4,
                            "gamma": 0.99,
                            "human_flag": False,
                            "max_episode_steps": 2,
                            "max_steps": 128,
                            "tau": 0.001,
                            "update_every": 4,
                            "num_updates": 1
                        }
                    },
                ]
            }

        return Configuration(test_flag=True, exp_cfg=cfg)
