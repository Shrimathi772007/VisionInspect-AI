"""Exception hierarchy for the image preprocessing pipeline.

Deliberately plain Python exceptions (not fastapi.HTTPException) so this
package stays usable outside of a request/response cycle. Callers that expose
this over an API can catch these and translate them to HTTP responses.
"""


class ImageProcessingError(Exception):
    """Base class for all preprocessing/quality-analysis errors."""


class ImageNotFoundError(ImageProcessingError):
    """The given path does not exist or is not a file."""


class UnsupportedImageFormatError(ImageProcessingError):
    """The file extension is not one of the project's supported image formats."""


class ImageDecodeError(ImageProcessingError):
    """The file exists and has a supported extension, but could not be decoded as an image."""
