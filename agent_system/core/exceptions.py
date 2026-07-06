"""Shared exception types."""


class TradingAgentError(Exception):
    """Base exception for trading agent failures."""


class StreamDataError(TradingAgentError):
    """Raised when market stream data is malformed or unusable."""
