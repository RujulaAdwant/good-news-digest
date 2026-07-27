"""Domain-specific exceptions for pipeline stages."""


class DuplicateDetectionError(Exception):
    """Raised when embedding-based deduplication fails."""


class SentimentScoringError(Exception):
    """Raised when sentiment classification fails."""


class SummarizationError(Exception):
    """Raised when Claude summarization fails after retries."""
