"""Exceptions for AI inference (single-image prediction).

Plain Python exceptions, consistent with app.ai.preprocessing.errors and
app.ai.training.errors - this package stays usable outside a request/response
cycle. Messages describe the category/model involved, not raw filesystem
paths, so they stay safe to surface to a caller further up the stack (e.g. a
future inspection API).
"""


class InferenceError(Exception):
    """Base class for all AI inference errors."""


class ModelArtifactNotFoundError(InferenceError):
    """No trained model artifact exists for the requested category/model name.

    Covers both an unsupported category and a valid category whose artifact
    is missing on disk - callers only need to know inference isn't possible,
    not the underlying filesystem reason.
    """


class ModelIntegrityError(InferenceError):
    """The configured model artifact exists but is not the validated model it is configured as.

    Raised instead of serving a replaced/retrained file under a validated threshold.
    """
