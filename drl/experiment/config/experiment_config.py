from typing import List, Dict
import json

from drl.experiment.config.agent_config import AgentConfig
from drl.experiment.config.environment_config import EnvironmentConfig
from drl.experiment.config.neural_network_config import NeuralNetworkConfig
from drl.experiment.config.reinforcement_learning_config import ReinforcementLearningConfig
from drl.experiment.config.replay_memory_config import ReplayMemoryConfig
from drl.experiment.config.trainer_config import TrainerConfig


class ExperimentConfig(object):
    def __init__(self,
                 id: str,
                 gym_id: str,
                 agent_cfg: AgentConfig,
                 environment_cfg: EnvironmentConfig,
                 trainer_cfg: TrainerConfig,
                 reinforcement_learning_cfg: ReinforcementLearningConfig,
                 replay_memory_cfg: ReplayMemoryConfig
                 ):
        self.id = id
        self.gym_id = gym_id
        self.agent_cfg = agent_cfg
        self.environment_cfg = environment_cfg
        self.trainer_cfg = trainer_cfg
        self.reinforcement_learning_cfg = reinforcement_learning_cfg
        self.replay_memory_cfg = replay_memory_cfg

    @classmethod
    def from_json(cls, data):
        data['agent_cfg'] = AgentConfig.from_json(data['agent_cfg'])
        data['environment_cfg'] = EnvironmentConfig.from_json(data['environment_cfg'])
        data['trainer_cfg'] = TrainerConfig.from_json(data['trainer_cfg'])
        data['reinforcement_learning_cfg'] = ReinforcementLearningConfig.from_json(data['reinforcement_learning_cfg'])
        data['replay_memory_cfg'] = ReplayMemoryConfig.from_json(data['replay_memory_cfg'])
        return cls(**data)



