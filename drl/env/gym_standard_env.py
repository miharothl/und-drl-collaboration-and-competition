import gym
from gym.spaces import Discrete

from drl import drl_logger
from drl.env.i_environment import IEnvironment


class GymStandardEnv(IEnvironment):

    def __init__(self, name):
        self.__env = gym.make(name)
        self.__env.seed(0)
        self.__termination_reward = 0

    def action_offset(self):
        return 0

    def close(self):
        self.__env.close()

    def get_action_space(self):

        isDiscrete = isinstance(self.__env.action_space, Discrete)

        if isDiscrete:
            num_action_space = self.__env.action_space.n
            drl_logger.info("Env action space is discrete")
            drl_logger.info("Env action space: {}".format(num_action_space))

        drl_logger.info("Env observation space: {}".format(self.__env.observation_space))


    def render(self, mode):
        self.__env.render(mode=mode)

    def reset(self):

        new_life = True
        return self.__env.reset(), new_life

    def start_game_action(self):
        return None

    def step(self, action):
        next_state, reward, done, info = self.__env.step(action)

        if done:
            reward += self.__termination_reward

        new_life = False

        return next_state, reward, done, new_life
