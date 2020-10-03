from drl.experiment.config.config_base import ConfigBase
from drl.experiment.config.neural_network_config import NeuralNetworkConfig


class DqnConfig(ConfigBase):
    def __init__(self,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay,
                 lr,
                 model_cfg : NeuralNetworkConfig
                 ):

        self.ensure_betwen_0_and_1(epsilon_start)
        self.ensure_betwen_0_and_1(epsilon_end)
        self.ensure_betwen_0_and_1(epsilon_decay)
        self.ensure_betwen_0_and_1(lr)

        self.ensure_is_greater(param1=epsilon_start, param2=epsilon_end)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.model_cfg = model_cfg

    @classmethod
    def from_json(cls, data):

        data['model_cfg'] = NeuralNetworkConfig.from_json(data['model_cfg'])

        return cls(**data)
