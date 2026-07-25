"""Domain-specific errors with safe messages for the user interface."""


class PegasusError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(PegasusError):
    """Invalid or missing runtime configuration."""


class DocumentError(PegasusError):
    """A document could not be validated or parsed."""


class UnsupportedFormatError(DocumentError):
    """The uploaded file type is outside the accepted formats."""


class EmptyDocumentError(DocumentError):
    """The document contains no usable text."""


class EncryptedDocumentError(DocumentError):
    """The PDF is password protected."""


class ScannedDocumentError(DocumentError):
    """The PDF appears to contain images without embedded text."""


class CorruptDocumentError(DocumentError):
    """The document structure could not be parsed."""


class GenerationError(PegasusError):
    """The language model could not generate an answer."""


class MissingApiKeyError(GenerationError):
    """No Gemini API key is configured."""


class QuotaExceededError(GenerationError):
    """The free Gemini quota is exhausted."""


class ProviderUnavailableError(GenerationError):
    """Gemini is temporarily unavailable or unreachable."""

