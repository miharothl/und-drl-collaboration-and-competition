from drl.experiment.configuration import Configuration
from drl.experiment.train.master_trainer import MasterTrainer
from drl.experiment.train.i_trainer import ITrainer


class TrainerFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(cfg: Configuration, session_id) -> ITrainer:

        algorithm_type = cfg.get_current_exp_cfg().reinforcement_learning_cfg.algorithm_type

        if algorithm_type.startswith('dqn'):
            trainer = MasterTrainer(cfg=cfg, session_id=session_id)
        elif algorithm_type.startswith('ddpg'):
            trainer = MasterTrainer(cfg=cfg, session_id=session_id)
        else:
            raise Exception("Trainer for algorighm '{}' type not supported".format(algorithm_type))

        return trainer
