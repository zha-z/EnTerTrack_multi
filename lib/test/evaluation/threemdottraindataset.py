from pathlib import Path

from lib.test.evaluation.data import BaseDataset
from lib.test.evaluation.threemdottestdataset import ThreemdotDataset


class ThreemdotTrainDataset(ThreemdotDataset):
    """ThreeMDOT train split exposed to the test runner for offline analysis."""

    def __init__(self):
        BaseDataset.__init__(self)
        self.base_path = self.env_settings.threemdot_val_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def _get_sequence_list(self):
        split_file = (
            Path(__file__).resolve().parents[2]
            / "train"
            / "data_specs"
            / "threemdot"
            / "threemdot_train.txt"
        )
        with split_file.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
