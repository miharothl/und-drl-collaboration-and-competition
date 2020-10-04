from unityagents import UnityEnvironment

from drl import drl_logger
from drl.env.i_environment import IEnvironment


class UnityEnv(IEnvironment):

    def __init__(self, name):

        drl_logger.info(
            "Initializing environment.'",
            extra={"params": {
                "name": name,
            }})

        self.env = UnityEnvironment(file_name=name)
        self.brain_name = self.env.brain_names[0]
        self.termination_reward = 0

    def action_offset(self):
        return 0

    def close(self):
        self.env.close()

    def get_action_space(self):
        # isDiscrete = isinstance(self.__env.action_space, Discrete)
        #
        # if isDiscrete:
        #     num_action_space = self.__env.action_space.n
        #     logging.debug("Env action space is discrete")
        #     logging.debug("Env action space: {}".format(num_action_space))
        #
        # logging.debug("Env observation space: {}".format(self.__env.observation_space))
        pass

    def render(self, mode):
        pass

    def reset(self):
        brain_name = self.env.brain_names[0]
        # brain = self.__env.brains[brain_name]

        env_info = self.env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        # state = env_info.vector_observations  # get the current state

        new_life = True

        return state, new_life

    def start_game_action(self):
        return None

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]  # send the action to the environment

        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished

        if done:
            reward += self.termination_reward

        new_life = False

        return next_state, reward, done, new_life
