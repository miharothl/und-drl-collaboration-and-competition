from collections import namedtuple

import numpy as np
import random
import torch
from torch import Tensor
import torch.optim as optim

from drl.agent.agent import Agent
from drl.agent.tools.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from drl.agent.tools.schedules import LinearSchedule
from drl.experiment.configuration import Configuration
from drl.model.model_factory import ModelFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DqnAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, seed, cfg: Configuration):

        super(DqnAgent, self).__init__(cfg)
        """Initialize an Agent object.

        Params
        ======
            seed (int): random seed
            cfg (Config): configration
        """
        self.double_dqn = False
        if self.reinforcement_learning_cfg.algorithm_type == 'dqn_double':
            self.double_dqn = True

        # Q-Network
        self.current_model, self.target_model = ModelFactory.create(seed, device, cfg)
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=self.reinforcement_learning_cfg.dqn_cfg.lr)

        # Replay Memory
        if self.replay_memory_cfg.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(int(self.replay_memory_cfg.buffer_size), alpha=self.replay_memory_cfg.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = self.trainer_cfg.max_steps
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                initial_p=self.replay_memory_cfg.prioritized_replay_beta0,
                                                final_p=1.0)
        else:
            self.memory = ReplayBuffer(self.replay_memory_cfg.buffer_size)
            self.beta_schedule = None

        # Initialize time step counter (for prioritized memory replay)
        self.step_counter = 0
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.step_update_counter = 0

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.current_model.eval()
        with torch.no_grad():
            action_values = self.current_model(state)
        self.current_model.train()

        eps = min(eps, 1.)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.agent_cfg.action_size))

    def get_models(self):

        model = namedtuple('name', 'weights')
        model.name = 'current'
        model.weights = self.current_model

        return [model]

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights = experiences

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1).to(device)
        weights = torch.from_numpy(weights).float().unsqueeze(1).to(device)

        if self.double_dqn:
            q_values = self.current_model(states)
            next_q_values = self.current_model(next_states)
            next_q_state_values = self.target_model(next_states)

            q_value = q_values.gather(1, actions).squeeze(1)

            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        else:  # dqn, dueling
            q_values = self.current_model(states)
            next_q_values = self.target_model(next_states)

            q_value = q_values.gather(1, actions).squeeze(1)
            next_q_value = next_q_values.max(1)[0]

        q_value = q_value.unsqueeze(1)
        next_q_value = next_q_value.unsqueeze(1)

        expected_q_value = rewards + gamma * next_q_value * (1 - dones)

        loss = (q_value - expected_q_value).pow(2) * weights

        td_error = loss
        td_error = td_error.squeeze(1)
        td_error = Tensor.cpu(td_error.detach())
        td_error = td_error.numpy()

        loss = loss.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.current_model, self.target_model, self.trainer_cfg.tau)

        return float(torch.sum(rewards > 0)) / rewards.shape[0], float(torch.sum(rewards < 0)) / rewards.shape[
            0], loss.item(), td_error

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        pos_reward_ratio = None
        neg_reward_ratio = None
        loss = None
        beta = None

        # Learn every UPDATE_EVERY time steps.
        self.step_update_counter = (self.step_update_counter + 1) % self.trainer_cfg.update_every
        if self.step_update_counter == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.trainer_cfg.batch_size:

                if self.replay_memory_cfg.prioritized_replay:
                    beta = self.beta_schedule.value(self.step_counter)
                    experience = self.memory.sample(self.trainer_cfg.batch_size, beta=beta)
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    exp = (obses_t, actions, rewards, obses_tp1, dones, weights)
                else:
                    experiences = self.memory.sample(self.trainer_cfg.batch_size)
                    obses_t, actions, rewards, obses_tp1, dones = experiences
                    weights, batch_idxes = np.ones_like(rewards), None
                    exp = (obses_t, actions, rewards, obses_tp1, dones, weights)

                pos_reward_ratio, neg_reward_ratio, loss, td_error = self.learn(exp, self.trainer_cfg.gamma)

                if self.replay_memory_cfg.prioritized_replay:
                    new_priorities = np.abs(td_error) + self.replay_memory_cfg.prioritized_replay_eps
                    self.memory.update_priorities(batch_idxes, new_priorities)

        self.step_counter += 1

        return pos_reward_ratio, neg_reward_ratio, loss, beta
