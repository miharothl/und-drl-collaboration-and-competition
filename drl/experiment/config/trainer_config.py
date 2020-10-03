from drl.experiment.config.config_base import ConfigBase


class TrainerConfig(ConfigBase):
    def __init__(self,
                 max_steps,
                 max_episode_steps,
                 eval_frequency,
                 eval_steps,
                 human_flag,
                 batch_size,
                 update_every,
                 num_updates,
                 tau,
                 gamma,
                 ):

        self.ensure_betwen_0_and_1(tau)
        self.ensure_betwen_0_and_1(gamma)

        self.max_steps = max_steps
        self.max_episode_steps = max_episode_steps
        self.eval_frequency = eval_frequency
        self.eval_steps = eval_steps
        self.human_flag = human_flag
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_updates = num_updates
        self.tau = tau
        self.gamma = gamma
