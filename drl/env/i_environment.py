import abc


class IEnvironment(abc.ABC):

    @abc.abstractmethod
    def action_offset(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def get_action_space(self):
        pass

    @abc.abstractmethod
    def action_offset(self):
        pass

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def start_game_action(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass
