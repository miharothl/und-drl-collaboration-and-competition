import gym
from gym.spaces import Discrete
from unityagents import UnityEnvironment

from drl import logging
from drl.env.i_environment import IEnvironment
from drl.env.unity_env import UnityEnv


class UnityMultipleEnv(UnityEnv):

    def reset(self):
        brain_name = self.env.brain_names[0]
        # brain = self.__env.brains[brain_name]

        # env_info = self.env.reset(train_mode=True)[brain_name]  # reset the environment
        env_info = self.env.reset(train_mode=False)[brain_name]  # reset the environment
        # state = env_info.vector_observations[0]  # get the current state
        state = env_info.vector_observations  # get the current state

        new_life = True

        return state, new_life

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]  # send the action to the environment

        next_state = env_info.vector_observations  # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished

        for i in range(len(reward)):
            if done[i]:
                reward[i] += self.termination_reward

        new_life = False

        return next_state, reward, done, new_life
