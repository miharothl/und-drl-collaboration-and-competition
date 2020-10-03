import os
import re
from pathlib import Path

from drl.agent.i_agent import IAgent
from drl.env.i_environment import IEnvironment
from drl.experiment.configuration import Configuration
from drl.experiment.train.i_trainer import ITrainer


class Trainer(ITrainer):

    def __init__(self, config: Configuration, session_id, path_models='models'):
        self.cfg = config
        self.session_id = session_id

    def get_model_filename(self, name, episode, score, val_score, eps):

        session_path = os.path.join(self.cfg.get_app_experiments_path(train_mode=True), self.session_id)
        Path(session_path).mkdir(parents=True, exist_ok=True)

        model_id = re.sub('[^0-9a-zA-Z]+', '', self.cfg.get_current_exp_cfg().id)
        model_id = model_id.lower()
        filename = "{}_{}_{}_{}_{:.2f}_{:.2f}_{:.2f}.pth".format(
            model_id, name, self.session_id, episode, score, val_score, eps)

        model_path = os.path.join(session_path, filename)

        return model_path

    def select_model_filename(self, model_filename=None):
        if model_filename is not None:
            path = os.path.join(self.__path_models, model_filename)
            return path

    def train(self, agent: IAgent, env: IEnvironment, model_filename=None):
        pass
