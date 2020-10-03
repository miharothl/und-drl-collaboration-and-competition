from drl.experiment.config.config_base import ConfigBase


class ReplayMemoryConfig(ConfigBase):
    def __init__(self,
                 buffer_size:int,
                 prioritized_replay:bool,
                 prioritized_replay_alpha:float,
                 prioritized_replay_beta0:float,
                 prioritized_replay_eps:float,
                 ):

            # self.ensure_betwen_0_and_1(prioritized_replay_alpha)
            # self.ensure_betwen_0_and_1(prioritized_replay_beta0)
            # self.ensure_betwen_0_and_1(prioritized_replay_eps)

            self.buffer_size = buffer_size
            self.prioritized_replay = prioritized_replay
            self.prioritized_replay_alpha = prioritized_replay_alpha
            self.prioritized_replay_beta0 = prioritized_replay_beta0
            self.prioritized_replay_eps = prioritized_replay_eps

