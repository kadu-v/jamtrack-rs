import unittest
from pathlib import Path

from generate_mcbyte_data import encode_rle, pending_frame_paths


class GenerateMcByteDataTests(unittest.TestCase):
    def test_resume_compares_completed_frame_ids_not_list_indices(self) -> None:
        paths = [Path("frame_000100.jpg"), Path("frame_000101.jpg")]
        self.assertEqual(
            pending_frame_paths(paths, {100}),
            [Path("frame_000101.jpg")],
        )

    def test_rle_starts_with_a_zero_run_for_foreground_first_masks(self) -> None:
        self.assertEqual(encode_rle([1, 1, 0, 0]), [0, 2, 2])


if __name__ == "__main__":
    unittest.main()
