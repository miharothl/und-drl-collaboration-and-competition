import os
from typing import List

from drl.experiment.config.experiment_config import ExperimentConfig
from drl.experiment.config.master_config import MasterConfig


class Configuration:
    def __init__(self, current_exp='lunarlander', test_flag=False, exp_cfg=None):
        self.__app = self.__set_app_config()
        self.__exp_cfg = self.__set_exp_config(exp_cfg)

        self.__exp_cfg.set_current(current_exp)

        self.__test = test_flag

    def get_app_config(self):
        return self.__app

    def get_exp_ids(self) -> List[str]:
        return self.__exp_cfg.get_ids()

    def set_current_exp_cfg(self, exp_id):
        self.__exp_cfg.set_current(exp_id)

    def get_current_exp_cfg(self) -> ExperimentConfig:
        return self.__exp_cfg.get_current()

    # app
    def get_app_analysis_path(self, train_mode=True):

        if self.__test:
            if train_mode:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], 'analysis')
            else:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], 'analysis')
        else:
            if train_mode:
                return os.path.join(self.__app['path_experiments'], 'analysis')
            else:
                return os.path.join(self.__app['path_experiments'], 'analysis')

    def get_app_experiments_path(self, train_mode=True):

        if self.__test:
            if train_mode:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], self.__app['path_train'])
            else:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], self.__app['path_play'])
        else:
            if train_mode:
                return os.path.join(self.__app['path_experiments'], self.__app['path_train'])
            else:
                return os.path.join(self.__app['path_experiments'], self.__app['path_play'])

    def __set_app_config(self):
        return {
            'path_experiments': '_experiments',
            'path_tests': '_tests',
            'path_play': 'play',
            'path_train': 'train',
        }

    def __set_exp_config(self, cfg) -> MasterConfig:

        if cfg is not None:
            return MasterConfig.from_json(cfg)

        cfg = {
            "experiment_cfgs": [
                ########################################################################################################
                # gym
                ########################################################################################################
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
                            "epsilon_start": 1,
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
                {
                    "id": "walker",
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
                            "lr_actor": 1e-04,
                            "lr_critic": 3e-04,
                            "weight_decay":  0,
                            "actor_model_cfg": {
                                "hidden_layers": [
                                    256,
                                    128
                                ]
                            },
                            "critic_model_cfg": {
                                "hidden_layers": [
                                    256,
                                    128
                                ]
                            },
                        }
                    },
                    "replay_memory_cfg": {
                        "buffer_size": 1e05,
                        "prioritized_replay": False,
                        "prioritized_replay_alpha": 0.6,
                        "prioritized_replay_beta0": 0.4,
                        "prioritized_replay_eps": 1e-06
                    },
                    "trainer_cfg": {
                        "batch_size": 128,
                        "eval_frequency": 20000,
                        "eval_steps": 3000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 700,
                        "max_steps": 1e06,
                        "tau": 1e-3,
                        "update_every": 1,
                        "num_updates": 1
                    }
                },
                {
                    "id": "pendulum",
                    "gym_id": "Pendulum-v0",
                    "agent_cfg": {
                        "action_size": 1,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 3
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
                            "lr_actor": 1e-04,
                            "lr_critic": 1e-03,
                            "weight_decay":  0.0001,
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
                        "buffer_size": 1e06,
                        "prioritized_replay": False,
                        "prioritized_replay_alpha": 0.6,
                        "prioritized_replay_beta0": 0.4,
                        "prioritized_replay_eps": 1e-06
                    },
                    "trainer_cfg": {
                        "batch_size": 128,
                        "eval_frequency": 20000,
                        "eval_steps": 3000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 300,
                        "max_steps": 1e06,
                        "tau": 1e-3,
                        "update_every": 1,
                        "num_updates": 1
                    }
                },
                ########################################################################################################
                # atari
                ########################################################################################################
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
                        "algorithm_type": "dqn_dueling",

                        "dqn_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.1,
                            "epsilon_decay": 0.99995,
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
                        "buffer_size": 1e06,
                        "prioritized_replay": True,
                        "prioritized_replay_alpha": 0.6,
                        "prioritized_replay_beta0": 0.4,
                        "prioritized_replay_eps": 1e-06
                    },
                    "trainer_cfg": {
                        "batch_size": 256,
                        "eval_frequency": 20000,
                        "eval_steps": 3000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 1000,
                        "max_steps": 50e6,
                        "tau": 0.001,
                        "update_every": 100,
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

                ########################################################################################################
                # unity
                ########################################################################################################
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
                            "epsilon_start": 1,
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
                        "eval_frequency": 10200,
                        "eval_steps": 2100,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 1000,
                        "max_steps": 600000,
                        "tau": 0.001,
                        "update_every": 4,
                        "num_updates": 1
                    }
                },
                {
                    'id': 'reacher',
                    'gym_id': 'env/unity/mac/reacher-single-agent.app',
                    "agent_cfg": {
                        "action_size": 4,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 33
                    },
                    "environment_cfg": {
                        "env_type": "unity",
                        "num_agents": 1
                    },
                    "reinforcement_learning_cfg": {
                        "algorithm_type": "ddpg",
                        "dqn_cfg": None,
                        "ddpg_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.01,
                            "epsilon_decay": 0.995,
                            "lr_actor": 1e-04,
                            "lr_critic": 1e-03,
                            "weight_decay":  0,
                            "actor_model_cfg": {
                                "hidden_layers": [
                                    256,
                                    128
                                ]
                            },
                            "critic_model_cfg": {
                                "hidden_layers": [
                                    256,
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
                        "batch_size": 128,
                        "eval_frequency": 10000,
                        "eval_steps": 2100,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 1000,
                        "max_steps": 300000,
                        "tau": 0.001,
                        "update_every": 1,
                        "num_updates": 1
                    }
                },
                {
                    'id': 'reacher-multiple',
                    'gym_id': 'env/unity/mac/reacher-multiple-agent.app',
                    "agent_cfg": {
                        "action_size": 4,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 33
                    },
                    "environment_cfg": {
                        "env_type": "unity-multiple",
                        "num_agents": 20
                    },
                    "reinforcement_learning_cfg": {
                        "algorithm_type": "ddpg",
                        "dqn_cfg": None,
                        "ddpg_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.1,
                            "epsilon_decay": 0.97,
                            "lr_actor": 1e-04,
                            "lr_critic": 3e-04,
                            "weight_decay":  0,
                            "actor_model_cfg": {
                                "hidden_layers": [
                                    256,
                                    128
                                ]
                            },
                            "critic_model_cfg": {
                                "hidden_layers": [
                                    256,
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
                        "batch_size": 128,
                        "eval_frequency": 10000,
                        "eval_steps": 2000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 1000,
                        "max_steps": 300000,
                        "tau": 0.001,
                        "update_every": 10,
                        "num_updates": 20
                    }
                },
                {
                    'id': 'reacher-multiple-linux',
                    'gym_id': 'env/unity/linux/reacher-multiple-agent-novis/Reacher.x86_64',
                    "agent_cfg": {
                        "action_size": 4,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 33
                    },
                    "environment_cfg": {
                        "env_type": "unity-multiple",
                        "num_agents": 20
                    },
                    "reinforcement_learning_cfg": {
                        "algorithm_type": "ddpg",
                        "dqn_cfg": None,
                        "ddpg_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.1,
                            "epsilon_decay": 0.97,
                            "lr_actor": 1e-04,
                            "lr_critic": 3e-04,
                            "weight_decay":  0,
                            "actor_model_cfg": {
                                "hidden_layers": [
                                    256,
                                    128
                                ]
                            },
                            "critic_model_cfg": {
                                "hidden_layers": [
                                    256,
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
                        "batch_size": 128,
                        "eval_frequency": 10000,
                        "eval_steps": 2000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 1000,
                        "max_steps": 300000,
                        "tau": 0.001,
                        "update_every": 10,
                        "num_updates": 20
                    }
                },
                {
                    'id': 'reacher-linux',
                    'gym_id': 'env/unity/linux/reacher-single-agent-novis/Reacher.x86_64',
                    "agent_cfg": {
                        "action_size": 4,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 33
                    },
                    "environment_cfg": {
                        "env_type": "unity",
                        "num_agents": 1
                    },
                    "reinforcement_learning_cfg": {
                        "algorithm_type": "ddpg",
                        "dqn_cfg": None,
                        "ddpg_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.01,
                            "epsilon_decay": 0.995,
                            "lr_actor": 1e-04,
                            "lr_critic": 1e-03,
                            "weight_decay":  0,
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
                        "batch_size": 128,
                        "eval_frequency": 10000,
                        "eval_steps": 2100,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 1000,
                        "max_steps": 600000,
                        "tau": 0.001,
                        "update_every": 4,
                        "num_updates": 1
                    }
                },
                {
                    'id': 'crawler',
                    'gym_id': 'env/unity/mac/crawler.app',
                    "agent_cfg": {
                        "action_size": 240,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 129
                    },
                    "environment_cfg": {
                        "env_type": "unity",
                        "num_agents": 1
                    },
                    "reinforcement_learning_cfg": {
                        "algorithm_type": "ddpg",
                        "dqn_cfg": None,
                        "ddpg_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.1,
                            "epsilon_decay": 0.9995,
                            "lr_actor": 1e-04,
                            "lr_critic": 3e-04,
                            "weight_decay":  0,
                            "actor_model_cfg": {
                                "hidden_layers": [
                                    1024,
                                    1024
                                ]
                            },
                            "critic_model_cfg": {
                                "hidden_layers": [
                                    1024,
                                    1024
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
                        "batch_size": 256,
                        "eval_frequency": 10000,
                        "eval_steps": 2000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 3000,
                        "max_steps": 3000000,
                        "tau": 0.001,
                        "update_every": 4,
                        "num_updates": 2
                    }
                },
                {
                    'id': 'crawler-linux',
                    'gym_id': 'env/unity/linux/crawler-novis/Crawler.x86_64',
                    "agent_cfg": {
                        "action_size": 240,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 129
                    },
                    "environment_cfg": {
                        "env_type": "unity",
                        "num_agents": 1
                    },
                    "reinforcement_learning_cfg": {
                        "algorithm_type": "ddpg",
                        "dqn_cfg": None,
                        "ddpg_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.1,
                            "epsilon_decay": 0.9997,
                            "lr_actor": 1e-06,
                            "lr_critic": 1e-06,
                            "weight_decay":  0,
                            "actor_model_cfg": {
                                "hidden_layers": [
                                    1024,
                                    1024
                                ]
                            },
                            "critic_model_cfg": {
                                "hidden_layers": [
                                    1024,
                                    1024
                                ]
                            },
                        }
                    },
                    "replay_memory_cfg": {
                        "buffer_size": 10000,
                        "prioritized_replay": False,
                        "prioritized_replay_alpha": 0.6,
                        "prioritized_replay_beta0": 0.4,
                        "prioritized_replay_eps": 1e-06
                    },
                    "trainer_cfg": {
                        "batch_size": 128,
                        "eval_frequency": 10000,
                        "eval_steps": 2000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 3000,
                        "max_steps": 3000000,
                        "tau": 0.001,
                        "update_every": 40,
                        "num_updates": 20
                    }
                },
                {
                    'id': 'tennis',
                    'gym_id': 'env/unity/mac/tennis.app',
                    "agent_cfg": {
                        "action_size": 2,
                        "discrete": True,
                        "num_frames": 1,
                        "state_rgb": False,
                        "state_size": 24
                    },
                    "environment_cfg": {
                        "env_type": "unity-multiple",
                        "num_agents": 2
                    },
                    "reinforcement_learning_cfg": {
                        "algorithm_type": "ddpg",
                        "dqn_cfg": None,
                        "ddpg_cfg": {
                            "epsilon_start": 1.0,
                            "epsilon_end": 0.1,
                            "epsilon_decay": 0.9995,
                            "lr_actor": 1e-04,
                            "lr_critic": 3e-04,
                            "weight_decay":  0,
                            "actor_model_cfg": {
                                "hidden_layers": [
                                    256,
                                    128
                                ]
                            },
                            "critic_model_cfg": {
                                "hidden_layers": [
                                    256,
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
                        "batch_size": 128,
                        "eval_frequency": 10000,
                        "eval_steps": 2000,
                        "gamma": 0.99,
                        "human_flag": False,
                        "max_episode_steps": 1000,
                        "max_steps": 300000,
                        "tau": 0.001,
                        "update_every": 2,
                        "num_updates": 4
                    }
                },


            ]
        }

        return MasterConfig.from_json(cfg)
