"""Domain-specific exceptions for pipeline stages."""


class DuplicateDetectionError(Exception):
    """Raised when embedding-based deduplication fails."""


class SentimentScoringError(Exception):
    """Raised when sentiment classification fails."""


class SummarizationError(Exception):
    """Raised when Claude summarization fails after retries."""


class DigestSelectionError(Exception):
    """Raised when digest candidate selection fails."""


class RelevanceFilterError(Exception):
    """Raised when good-news vs corporate embedding relevance filtering fails."""


class DigestCompileError(Exception):
    """Raised when compiling/persisting a daily digest fails."""


class DigestResetError(Exception):
    """Raised when unlocking a digest for re-send is refused or fails."""


class EmailDeliveryError(Exception):
    """Raised when formatting or sending the digest email fails."""
