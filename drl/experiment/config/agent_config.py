from drl.experiment.config.config_base import ConfigBase


class AgentConfig(ConfigBase):
    def __init__(self,
                 action_size: int,
                 state_size: int,
                 discrete: bool,
                 state_rgb: bool,
                 num_frames: int,
                 ):
        self.action_size = action_size
        self.state_size = state_size
        self.discrete = discrete
        self.state_rgb = state_rgb
        self.num_frames = num_frames

