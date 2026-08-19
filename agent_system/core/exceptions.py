"""Shared exception types."""


class TradingAgentError(Exception):
    """Base exception for trading agent failures."""


class StreamDataError(TradingAgentError):
    """Raised when market stream data is malformed or unusable."""


class MarketDataError(TradingAgentError):
    """Raised when historical or reference market data cannot be obtained."""


class ConfigurationError(TradingAgentError):
    """Raised when configuration is missing or inconsistent."""


class RiskLimitError(TradingAgentError):
    """Raised when an action would breach a hard risk limit."""


class ExecutionError(TradingAgentError):
    """Raised when order placement or account access fails."""
