import os
from pathlib import Path

from lib.test.evaluation.data import BaseDataset
from lib.test.evaluation.threemdottestdataset import ThreemdotDataset


class ThreemdotCVDataset(ThreemdotDataset):
    """ThreeMDOT train target-group CV split for held-out fold evaluation.

    The split file must be supplied through THREEMDOT_CV_SPLIT_FILE. This avoids
    adding many static dataset names and prevents accidental fallback to
    Three-MDOT val/test.
    """

    def __init__(self):
        BaseDataset.__init__(self)
        self.base_path = self.env_settings.threemdot_val_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def _get_sequence_list(self):
        split_file = os.environ.get("THREEMDOT_CV_SPLIT_FILE", "")
        if not split_file:
            raise RuntimeError(
                "THREEMDOT_CV_SPLIT_FILE must point to a fold holdout split")
        path = Path(split_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
