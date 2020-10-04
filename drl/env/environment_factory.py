from drl.env.gym_atari_space_invaders_env import GymAtariSpaceInvadersEnv
from drl.env.i_environment import IEnvironment
from drl.env.gym_atari_env import GymAtariEnv
from drl.env.gym_standard_env import GymStandardEnv
from drl.env.unity_env import UnityEnv
from drl.env.unity_multiple_env import UnityMultipleEnv
from drl.experiment.configuration import Configuration


class EnvironmentFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(config: Configuration) -> IEnvironment:

        env_name = config.get_current_exp_cfg().gym_id

        env_type = config.get_current_exp_cfg().environment_cfg.env_type

        if env_type == 'gym':
            env = GymStandardEnv(name=env_name)
        elif env_type == 'atari_gym':
            env = GymAtariEnv(name=env_name)
        elif env_type == 'spaceinvaders_atari_gym':
            env = GymAtariSpaceInvadersEnv(name=env_name)
        elif env_type == 'unity':
            env = UnityEnv(name=env_name)
        elif env_type == 'unity-multiple':
            env = UnityMultipleEnv(name=env_name)
        else:
            raise Exception("Environment '{}' type not supported".format(env_type))

        return env
