import numpy as np
import torch
from collections import deque

from drl.agent.i_agent import IAgent
from drl.experiment.configuration import Configuration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(IAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, cfg: Configuration):

        # training parameters
        self.trainer_cfg = cfg.get_current_exp_cfg().trainer_cfg

        # agent parameters
        self.agent_cfg = cfg.get_current_exp_cfg().agent_cfg

        # replay memory parameters
        self.replay_memory_cfg = cfg.get_current_exp_cfg().replay_memory_cfg
        self.prioritized_replay_beta_iters = None

        # reinforcement learning parameters
        self.reinforcement_learning_cfg = cfg.get_current_exp_cfg().reinforcement_learning_cfg

        self.frames_queue = deque(maxlen=self.agent_cfg.num_frames)

    def act(self, state):
        pass

    def learn(self, experiences, gamma):
        pass

    def pre_process(self, raw_state):

        if self.agent_cfg.state_rgb is True:
            from drl.image import preprocess_image
            raw_state = preprocess_image(raw_state)

        if len(self.frames_queue) == 0:
            for i in range(self.agent_cfg.num_frames):
                self.frames_queue.append(raw_state)

        self.frames_queue.append(raw_state)

        if self.agent_cfg.state_rgb is True:
            state = np.stack(self.frames_queue)
        else:
            state = np.concatenate(self.frames_queue)

        return state

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def step(self, state, action, reward, next_state, done):
        pass
