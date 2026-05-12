from __future__ import annotations

import unittest

import pandas as pd

from veri_madenciligi.services.phase2_eda import deduplicate_dataset, safe_quantile_bands


class Phase2EdaHelpersTestCase(unittest.TestCase):
    def test_deduplicate_dataset_removes_exact_duplicates(self) -> None:
        dataset = pd.DataFrame(
            {
                "Age": [20, 20, 30],
                "PurchaseStatus": [0, 0, 1],
            }
        )

        deduplicated = deduplicate_dataset(dataset)

        self.assertEqual(len(deduplicated), 2)

    def test_safe_quantile_bands_returns_non_empty_band_labels(self) -> None:
        series = pd.Series([10, 20, 30, 40, 50])

        bands = safe_quantile_bands(series)

        self.assertEqual(len(bands), 5)
        self.assertTrue(all(isinstance(value, str) and value for value in bands.tolist()))


if __name__ == "__main__":
    unittest.main()