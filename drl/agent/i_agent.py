import abc


class IAgent(abc.ABC):

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def get_models(self):
        pass

    @abc.abstractmethod
    def learn(self, experiences, gamma):
        pass

    @abc.abstractmethod
    def pre_process(self, raw_state):
        pass

    @abc.abstractmethod
    def soft_update(self, local_model, target_model, tau):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass


