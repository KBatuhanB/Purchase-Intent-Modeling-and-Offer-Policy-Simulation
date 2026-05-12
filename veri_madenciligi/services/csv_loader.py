from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from ..core.exceptions import DatasetLoadError


class CsvDatasetLoader:
    def __init__(
        self,
        read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
    ) -> None:
        self._read_csv = read_csv

    def load(self, dataset_path: Path) -> pd.DataFrame:
        resolved_path = dataset_path.resolve()

        if not resolved_path.exists() or not resolved_path.is_file():
            raise DatasetLoadError(f"Veri dosyasi bulunamadi: {resolved_path}")

        try:
            return self._read_csv(resolved_path)
        except FileNotFoundError as error:
            raise DatasetLoadError(f"Veri dosyasi bulunamadi: {resolved_path}") from error
        except (ParserError, UnicodeDecodeError, EmptyDataError) as error:
            raise DatasetLoadError(
                f"Veri dosyasi okunurken bozuk veya beklenmeyen bir format ile karsilasildi: {resolved_path}"
            ) from error
        except OSError as error:
            raise DatasetLoadError(f"Veri dosyasi okunurken isletim sistemi hatasi alindi: {resolved_path}") from error
