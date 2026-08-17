import unittest
from collections import Counter, defaultdict

import common
from lib.train.data.fcvc_sampler import FCVCSampler


MANIFEST = (
    common.ROOT
    / "output/multi_agent_collaboration_clean/fcvc_manual_run/full_train_receiver_manifest.csv"
)


class FCVCOstrackSamplerTest(unittest.TestCase):
    def setUp(self):
        self.sampler = FCVCSampler(MANIFEST)

    def test_epoch_contract_and_determinism(self):
        first, first_contract = self.sampler.generate_epoch(1)
        second, second_contract = self.sampler.generate_epoch(1)
        next_epoch, next_contract = self.sampler.generate_epoch(2)
        self.assertEqual(first, second)
        self.assertEqual(first_contract, second_contract)
        self.assertNotEqual(first_contract["manifest_sha256"], next_contract["manifest_sha256"])
        self.assertEqual(len(first), 10008)
        self.assertEqual(len(next_epoch), 10008)
        self.assertEqual(first_contract["epoch_seed"], 42)
        self.assertEqual(next_contract["epoch_seed"], 43)

    def test_sync_interval_receiver_and_target_balance(self):
        rows, contract = self.sampler.generate_epoch(1)
        groups = defaultdict(list)
        for row in rows:
            groups[row["sync_group_id"]].append(row)
            self.assertLessEqual(
                abs(row["search_frame"] - row["template_frame"]), 200)
            self.assertFalse(row["uses_gt_in_student_input"])
            self.assertEqual(row["split"], "official_train")
        self.assertEqual(len(groups), 3336)
        for group in groups.values():
            self.assertEqual({row["receiver"] for row in group}, {"A", "B", "C"})
            self.assertEqual(len({row["template_frame"] for row in group}), 1)
            self.assertEqual(len({row["search_frame"] for row in group}), 1)
            self.assertEqual(len({row["target"] for row in group}), 1)
        counts = Counter(row["target"] for row in rows[::3])
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(counts, Counter(contract["target_group_counts"]))

    def test_six_rank_partition_has_exact_coverage(self):
        full, contract = self.sampler.generate_epoch(1)
        parts = []
        for rank in range(6):
            local = FCVCSampler(MANIFEST)
            local_contract = local.begin_epoch(1, rank=rank, world_size=6)
            self.assertEqual(len(local.rows), 1668)
            self.assertEqual(local_contract["local_group_count"], 556)
            self.assertEqual(Counter(row["receiver"] for row in local.rows),
                             Counter({"A": 556, "B": 556, "C": 556}))
            parts.extend(local.rows)
        self.assertEqual(parts, full)
        self.assertEqual(len({row["case_index"] for row in parts}), 10008)
        self.assertEqual(contract["manifest_sha256"],
                         self.sampler.generate_epoch(1)[1]["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
