import unittest

import common


class FCVCFeatureTapContractTest(unittest.TestCase):
    test_tap_and_replay_output_identity = common.legacy_model.FCVCContractTest.test_dense_tap_replay_identity
