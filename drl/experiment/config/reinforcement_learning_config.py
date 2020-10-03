from drl.experiment.config.config_base import ConfigBase
from drl.experiment.config.ddpg_config import DdpgConfig
from drl.experiment.config.dqn_config import DqnConfig


class ReinforcementLearningConfig(ConfigBase):
    def __init__(self,
                 algorithm_type: str,
                 dqn_cfg: DqnConfig,
                 ddpg_cfg: DdpgConfig,
                 ):

        self.ensure_in_list(param=algorithm_type, valid=['dqn', 'dqn_double', 'dqn_dueling', 'ddpg'])

        self.algorithm_type = algorithm_type
        self.dqn_cfg = dqn_cfg
        self.ddpg_cfg = ddpg_cfg

        self.ensure_exists(if_alg_startswith='dqn', algorithm_type=algorithm_type, cfg=self.dqn_cfg)
        self.ensure_exists(if_alg_startswith='ddpg', algorithm_type=algorithm_type, cfg=self.ddpg_cfg)

    @classmethod
    def from_json(cls, data):

        if data['dqn_cfg'] is None:
            data['dqn_cfg'] = None
        else:
            data['dqn_cfg'] = DqnConfig.from_json(data['dqn_cfg'])

        if data['ddpg_cfg'] is None:
            data['ddpg_cfg'] = None
        else:
            data['ddpg_cfg'] = DdpgConfig.from_json(data['ddpg_cfg'])

        return cls(**data)
