import numpy as np
from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from drl import drl_logger
from drl.agent.agent import Agent
from drl.agent.ddpg_agent import DdpgAgent
from drl.agent.tools.ou_noise import OUNoise, OUNoiseStandardNormal
from drl.agent.tools.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from drl.agent.tools.schedules import LinearSchedule
from drl.experiment.configuration import Configuration
from drl.model.model_factory import ModelFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MaddpgAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, seed, cfg: Configuration, num_agents=1):

        """Initialize an Agent object.

        Params
        ======
            seed (int): random seed
            cfg (Config): configration
            num_agents(int): number of agents
        """

        self.num_agents = cfg.get_current_exp_cfg().environment_cfg.num_agents

        self.state_size = cfg.get_current_exp_cfg().agent_cfg.state_size
        self.action_size = cfg.get_current_exp_cfg().agent_cfg.action_size

        self.agents = []

        for _ in range(self.num_agents):
            agent = DdpgAgent(seed, cfg)
            self.agents.append(agent)

    def act(self, state, eps=0., add_noise=True):

        actions=[]
        for i in range(self.num_agents):
            action = self.agents[i].act(state=state[i], eps=eps)
            actions.append(action)

        actions = np.asarray(actions)

        return actions

    def get_models(self):

        model_actor = namedtuple('name', 'weights')
        model_actor.name = 'current_actor'
        model_actor.weights = self.actor_current_model

        model_critic = namedtuple('name', 'weights')
        model_critic.name = 'current_critic'
        model_critic.weights = self.critic_current_model

        return [model_actor, model_critic]

    def learn(self, experiences, gamma):

        for i in range(self.num_agents):
            self.agents[i].learn(experiences[i], gamma)

        return 0, 0, 0, 0

    def pre_process(self, state):
        return state

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):

        for i in range(self.num_agents):
            self.agents[i].step(state[i], action[i],reward[i],next_state[i], done[i])

        return 0, 0, 0, 0

