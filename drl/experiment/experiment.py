from datetime import datetime

from drl.agent.agent_factory import AgentFactory
from drl.env.environment_factory import EnvironmentFactory
from drl.experiment.configuration import Configuration
from drl.experiment.player import Player
from drl.experiment.train.trainer_factory import TrainerFactory


class Experiment:
    def __init__(self, config: Configuration):
        self.__config = config
        self.__timestamp = datetime.now().strftime("%Y%m%dT%H%M")

    def get_timestamp(self):
        return self.__timestamp

    def get_session_id(self):
        return "{}-{}".format(
            self.__config.get_current_exp_cfg().id,
            self.get_timestamp()
        )

    def list_envs(self):
        envs = self.__config.get_exp_ids()
        for e in envs:
            print(e)

        return envs

    def play(self, mode, model, num_episodes=3, trained=True, num_steps=None):

        with Player(
                env=EnvironmentFactory.create(self.__config),
                agent=AgentFactory.create(self.__config),
                config=self.__config,
                session_id=self.get_session_id()) as player:

            return player.play(trained=trained,
                        mode=mode,
                        is_rgb=self.__config.get_current_exp_cfg().agent_cfg.state_rgb,
                        model_filename=model,
                        num_episodes=num_episodes,
                        num_steps=num_steps)

    def play_dummy(self, mode, model, num_episodes=3, num_steps=None):
        self.play(trained=False,
                  mode=mode,
                  model=model,
                  num_episodes=num_episodes,
                  num_steps=num_steps)

    def set_env(self, env):
        self.__config.set_current_exp_cfg(env)

    def train(self, model=None):

        trainer = TrainerFactory.create(
            cfg=self.__config,
            session_id=self.get_session_id()
        )

        return trainer.train(
            AgentFactory.create(self.__config),
            EnvironmentFactory.create(self.__config),
            model_filename=model
        )
