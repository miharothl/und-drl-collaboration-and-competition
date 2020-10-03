from drl.experiment.config.config_base import ConfigBase
from drl.experiment.config.neural_network_config import NeuralNetworkConfig


class DdpgConfig(ConfigBase):
    def __init__(self,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 actor_model_cfg : NeuralNetworkConfig,
                 critic_model_cfg : NeuralNetworkConfig
                 ):

        self.ensure_betwen_0_and_1(epsilon_start)
        self.ensure_betwen_0_and_1(epsilon_end)
        self.ensure_betwen_0_and_1(epsilon_decay)
        self.ensure_betwen_0_and_1(lr_actor)
        self.ensure_betwen_0_and_1(lr_critic)
        self.ensure_betwen_0_and_1(weight_decay)

        self.ensure_is_greater(param1=epsilon_start, param2=epsilon_end)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr_actor = lr_actor          # learning rate of the actor
        self.lr_critic = lr_critic        # learning rate of the critic
        self.weight_decay = weight_decay  # L2 weight decay
        self.actor_model_cfg = actor_model_cfg
        self.critic_model_cfg = critic_model_cfg

    @classmethod
    def from_json(cls, data):

        data['actor_model_cfg'] = NeuralNetworkConfig.from_json(data['actor_model_cfg'])
        data['critic_model_cfg'] = NeuralNetworkConfig.from_json(data['critic_model_cfg'])

        return cls(**data)
