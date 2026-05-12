from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_COLUMNS = (
    "Age",
    "Gender",
    "AnnualIncome",
    "NumberOfPurchases",
    "ProductCategory",
    "TimeSpentOnWebsite",
    "LoyaltyProgram",
    "DiscountsAvailed",
    "PurchaseStatus",
)

CATEGORICAL_COLUMNS = (
    "Gender",
    "ProductCategory",
    "LoyaltyProgram",
)

SAFE_NUMERIC_COLUMNS = (
    "Age",
    "AnnualIncome",
    "NumberOfPurchases",
    "TimeSpentOnWebsite",
)

LEAKY_OR_ANALYSIS_ONLY_COLUMNS = (
    "DiscountsAvailed",
)

NUMERIC_COLUMNS = (
    "Age",
    "AnnualIncome",
    "NumberOfPurchases",
    "TimeSpentOnWebsite",
    "DiscountsAvailed",
)

INTEGER_LIKE_COLUMNS = (
    "Age",
    "Gender",
    "NumberOfPurchases",
    "ProductCategory",
    "LoyaltyProgram",
    "DiscountsAvailed",
    "PurchaseStatus",
)

TARGET_COLUMN = "PurchaseStatus"
VALID_TARGET_VALUES = frozenset({0, 1})


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path
    dataset_path: Path
    artifacts_dir: Path = field(init=False)
    phase_1_dir: Path = field(init=False)
    phase_2_dir: Path = field(init=False)
    phase_3_dir: Path = field(init=False)
    phase_4_dir: Path = field(init=False)
    phase_5_dir: Path = field(init=False)
    phase_6_dir: Path = field(init=False)
    phase_7_dir: Path = field(init=False)
    phase_8_dir: Path = field(init=False)
    phase_9_dir: Path = field(init=False)
    phase_10_dir: Path = field(init=False)
    phase_11_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        resolved_root = self.root_dir.resolve()
        resolved_dataset = self.dataset_path.resolve()

        object.__setattr__(self, "root_dir", resolved_root)
        object.__setattr__(self, "dataset_path", resolved_dataset)
        object.__setattr__(self, "artifacts_dir", resolved_root / "artifacts")
        object.__setattr__(self, "phase_1_dir", resolved_root / "artifacts" / "phase_1")
        object.__setattr__(self, "phase_2_dir", resolved_root / "artifacts" / "phase_2")
        object.__setattr__(self, "phase_3_dir", resolved_root / "artifacts" / "phase_3")
        object.__setattr__(self, "phase_4_dir", resolved_root / "artifacts" / "phase_4")
        object.__setattr__(self, "phase_5_dir", resolved_root / "artifacts" / "phase_5")
        object.__setattr__(self, "phase_6_dir", resolved_root / "artifacts" / "phase_6")
        object.__setattr__(self, "phase_7_dir", resolved_root / "artifacts" / "phase_7")
        object.__setattr__(self, "phase_8_dir", resolved_root / "artifacts" / "phase_8")
        object.__setattr__(self, "phase_9_dir", resolved_root / "artifacts" / "phase_9")
        object.__setattr__(self, "phase_10_dir", resolved_root / "artifacts" / "phase_10")
        object.__setattr__(self, "phase_11_dir", resolved_root / "artifacts" / "phase_11")
        object.__setattr__(self, "logs_dir", resolved_root / "logs")

    @classmethod
    def from_dataset_path(cls, dataset_path: Path) -> "ProjectPaths":
        return cls(root_dir=dataset_path.resolve().parent, dataset_path=dataset_path.resolve())

    def ensure_output_directories(self) -> None:
        for directory in (
            self.artifacts_dir,
            self.phase_1_dir,
            self.phase_2_dir,
            self.phase_3_dir,
            self.phase_4_dir,
            self.phase_5_dir,
            self.phase_6_dir,
            self.phase_7_dir,
            self.phase_8_dir,
            self.phase_9_dir,
            self.phase_10_dir,
            self.phase_11_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
