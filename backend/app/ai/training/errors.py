"""Exceptions for MVTec AD dataset discovery.

Plain Python exceptions, consistent with app.ai.preprocessing.errors -
this package stays usable outside a request/response cycle.
"""


class TrainingDataError(Exception):
    """Base class for all dataset discovery/preparation errors."""


class DatasetRootNotFoundError(TrainingDataError):
    """The configured MVTec AD dataset root does not exist on disk."""


class CategoryNotFoundError(TrainingDataError):
    """The requested MVTec category does not exist under the dataset root."""
