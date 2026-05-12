class DataAnalysisError(Exception):
    """Proje genelinde anlamli hata hiyerarsisi saglar."""


class DatasetLoadError(DataAnalysisError):
    """Veri dosyasi okunamadiginda firlatilir."""


class SchemaValidationError(DataAnalysisError):
    """Zorunlu sema kosullari saglanmadiginda firlatilir."""


class ArtifactWriteError(DataAnalysisError):
    """Rapor veya artefakt disk'e yazilamadiginda firlatilir."""
