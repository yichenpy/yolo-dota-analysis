class AnalysisError(Exception):
    """Base exception for analysis errors."""


class ConfigurationError(AnalysisError):
    """Raised when paths or configuration values are invalid."""


class DependencyError(AnalysisError):
    """Raised when an optional runtime dependency is missing."""


class DatasetError(AnalysisError):
    """Raised when the dataset cannot be parsed."""


class InferenceError(AnalysisError):
    """Raised when model inference fails."""
