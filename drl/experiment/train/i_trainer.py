import abc

from drl.agent.i_agent import IAgent
from drl.env.i_environment import IEnvironment


class ITrainer(abc.ABC):

    @abc.abstractmethod
    def get_model_filename(self, episode, score, val_score, eps):
        pass

    @abc.abstractmethod
    def select_model_filename(self, model_filename=None):
        pass

    @abc.abstractmethod
    def train(self, agent: IAgent, env: IEnvironment, model_filename=None):
        pass
