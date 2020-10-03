from typing import Dict, List

from drl.experiment.config.experiment_config import ExperimentConfig


class MasterConfig(object):
    def __init__(self, experiment_cfgs: Dict[str, ExperimentConfig]):
        self.experiment_cfgs = experiment_cfgs
        self.current = None

    def set_current(self, id):
        count = 0
        for cfg in self.experiment_cfgs:
            if cfg.id == id:
                self.current = count
                break

            count += 1

    def get_current(self):
        return self.experiment_cfgs[self.current]

    def get_ids(self) -> List[str]:
        ids = []
        for i in range(len(self.experiment_cfgs)):
            ids.append(self.experiment_cfgs[i].id)
        return ids

    @classmethod
    def from_json(cls, data):
        experiment_cfgs = list(map(ExperimentConfig.from_json, data['experiment_cfgs']))
        return cls(experiment_cfgs)
