from __future__ import annotations

import unittest

import pandas as pd

from veri_madenciligi.services.phase3_preprocessing import (
    FeatureEngineeringTransformer,
    build_feature_role_matrix,
)
from veri_madenciligi.core.exceptions import SchemaValidationError


class Phase3PreprocessingHelpersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.role_matrix = build_feature_role_matrix()
        self.dataset = pd.DataFrame(
            {
                "Age": [25, 35, 45, 55],
                "Gender": [0, 1, 0, 1],
                "AnnualIncome": [30000.0, 60000.0, 90000.0, 120000.0],
                "NumberOfPurchases": [1, 5, 10, 15],
                "ProductCategory": [0, 1, 2, 3],
                "TimeSpentOnWebsite": [10.0, 20.0, 30.0, 40.0],
                "LoyaltyProgram": [0, 1, 0, 1],
            }
        )

    def test_feature_engineering_transformer_adds_expected_columns(self) -> None:
        transformer = FeatureEngineeringTransformer(self.role_matrix)
        transformer.fit(self.dataset)

        transformed = transformer.transform(self.dataset)

        self.assertIn("income_bucket", transformed.columns)
        self.assertIn("income_per_purchase_proxy", transformed.columns)
        self.assertIn("category_income_interaction", transformed.columns)
        self.assertTrue(set(transformed["high_time_low_history_flag"].unique()).issubset({0, 1}))

    def test_feature_engineering_transformer_requires_named_dataframe_columns(self) -> None:
        transformer = FeatureEngineeringTransformer(self.role_matrix)
        incomplete_dataset = self.dataset.drop(columns=["Age"])

        with self.assertRaises(SchemaValidationError):
            transformer.fit(incomplete_dataset)

    def test_feature_role_matrix_keeps_discounts_availed_outside_main_model(self) -> None:
        self.assertNotIn("DiscountsAvailed", self.role_matrix.base_input_features)
        self.assertIn("DiscountsAvailed", self.role_matrix.excluded_from_main_model)


if __name__ == "__main__":
    unittest.main()