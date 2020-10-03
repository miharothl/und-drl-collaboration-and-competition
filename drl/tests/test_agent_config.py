import pytest

from drl.experiment.configuration import Configuration


class TestAgentConfig:

    @staticmethod
    def get():
        cfg = \
            {
                "experiment_cfgs": [
                    {
                        "id": "lunarlander-dqn",
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
                        "id": "lunarlander-dqn-withframes",
                        "gym_id": "LunarLander-v2",
                        "agent_cfg": {
                            "action_size": 4,
                            "discrete": True,
                            "num_frames": 10,
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
                        "id": "lunarlander-dqn-noprio",
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
                            "prioritized_replay": False,
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
                        "id": "lunarlander-dqn-dueling",
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
                            "algorithm_type": "dqn_dueling",
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
                        "id": "lunarlander-dqn-double",
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
                    {
                        "id": "walker-ddpg",
                        "gym_id": "BipedalWalker-v3",
                        "agent_cfg": {
                            "action_size": 4,
                            "discrete": True,
                            "num_frames": 1,
                            "state_rgb": False,
                            "state_size": 24
                        },
                        "environment_cfg": {
                            "env_type": "gym",
                            "num_agents": 1
                        },
                        "reinforcement_learning_cfg": {
                            "algorithm_type": "ddpg",
                            "dqn_cfg": None,
                            "ddpg_cfg": {
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.01,
                                "epsilon_decay": 0.995,
                                "lr_actor": 0.0001,
                                "lr_critic": 0.0003,
                                "weight_decay": 0.0001,
                                "actor_model_cfg": {
                                    "hidden_layers": [
                                        64,
                                        64
                                    ]
                                },
                                "critic_model_cfg": {
                                    "hidden_layers": [
                                        128,
                                        128
                                    ]
                                },
                            }
                        },
                        "replay_memory_cfg": {
                            "buffer_size": 100000,
                            "prioritized_replay": False,
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
