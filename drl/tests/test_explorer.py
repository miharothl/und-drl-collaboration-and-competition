import pytest

from drl.experiment.configuration import Configuration
from drl.experiment.explorer import Explorer


class TestExplorer:

    @pytest.mark.depends(on=['test_play'])
    def test_listPlayExperiments_experimentsExist_returnsExperiments(self):

        config = Configuration(test_flag=True)
        explorer = Explorer(config = config)

        experiments = explorer.list_play_experiments()
        assert len(experiments) > 0

    @pytest.mark.depends(on=['test_train'])
    def test_listTrainExperiments_experimentsExist_returnsExperiments(self):

        config = Configuration(test_flag=True)
        explorer = Explorer(config = config)

        experiments = explorer.list_train_experiments()
        assert len(experiments) > 0

