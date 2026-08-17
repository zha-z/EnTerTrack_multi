import base64
import pathlib
import sys
import unittest
import zlib

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.models.entertrack.c3r import C3R, C3R_PACKET_BYTES  # noqa: E402


def decode(row, name, dtype, shape):
    raw = zlib.decompress(base64.b64decode(row[name].encode("ascii")))
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class RemoteInformationInstrumentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260719)
        self.model = C3R(variant="c1").eval()
        self.local = torch.randn(1, 256, 192)
        self.remote = torch.randn(1, 256, 192)
        self.local_response = torch.sigmoid(torch.randn(1, 1, 16, 16))
        remote_response = torch.sigmoid(torch.randn(1, 1, 16, 16))
        message = self.model.encoder(
            self.remote,
            remote_response,
            bbox=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            previous_bbox=torch.tensor([[0.4, 0.5, 0.2, 0.2]]),
            sender_ids=[1],
            sequence_hashes=[17],
            frame_ids=[1],
            timestamp_ms=[33],
        )[0]
        self.packet = self.model.codec.serialize(message)

    def run_collaboration(self, rich):
        return self.model.collaborate(
            self.local,
            self.local_response,
            [self.packet],
            receiver_id=0,
            sequence_hash=17,
            local_frame_id=1,
            local_timestamp_ms=33,
            frame_interval_ms=33,
            instrumentation=True,
            remote_information_diagnostics=rich,
        )

    def test_default_off_is_bitwise_identical(self):
        disabled = self.run_collaboration(False)
        enabled = self.run_collaboration(True)
        self.assertTrue(torch.equal(
            disabled["search_tokens"], enabled["search_tokens"]))
        self.assertEqual(len(self.packet), C3R_PACKET_BYTES)
        self.assertNotIn(
            "local_prompt_f16_zlib_b64",
            disabled["instrumentation_source_rows"][0],
        )

    def test_rich_payload_shapes_and_finite_statistics(self):
        row = self.run_collaboration(True)["instrumentation_source_rows"][0]
        local_prompt = decode(
            row, "local_prompt_f16_zlib_b64", np.float16, (4, 64))
        remote_prompt = decode(
            row, "remote_prompt_f16_zlib_b64", np.float16, (4, 64))
        residual = decode(
            row, "adapted_residual_f16_zlib_b64", np.float16,
            tuple(row["adapted_residual_shape"]))
        channel_mean = decode(
            row, "adapted_residual_channel_mean_f16_zlib_b64",
            np.float16, (192,))
        self.assertEqual(local_prompt.shape, (4, 64))
        self.assertEqual(remote_prompt.shape, (4, 64))
        self.assertEqual(residual.shape, (1, 256, 192))
        self.assertEqual(channel_mean.shape, (192,))
        self.assertTrue(np.isfinite(residual).all())
        self.assertTrue(np.isfinite(channel_mean).all())
        self.assertEqual(row["remote_information_diagnostics"], True)

    def test_rich_requires_instrumentation(self):
        with self.assertRaisesRegex(ValueError, "require C3R instrumentation"):
            self.model.collaborate(
                self.local,
                self.local_response,
                [self.packet],
                receiver_id=0,
                sequence_hash=17,
                local_frame_id=1,
                local_timestamp_ms=33,
                frame_interval_ms=33,
                remote_information_diagnostics=True,
            )


if __name__ == "__main__":
    unittest.main()
