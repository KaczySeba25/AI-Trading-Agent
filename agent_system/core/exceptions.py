"""Shared exception types."""


class TradingAgentError(Exception):
    """Base exception for trading agent failures."""


class StreamDataError(TradingAgentError):
    """Raised when market stream data is malformed or unusable."""


class MarketDataError(TradingAgentError):
    """Raised when historical or reference market data cannot be obtained."""


class ConfigurationError(TradingAgentError):
    """Raised when the configuration is incomplete or contradictory."""


class ExecutionError(TradingAgentError):
    """Raised when an order cannot be placed or validated."""


class RiskLimitError(TradingAgentError):
    """Raised when an action would breach a risk limit."""
