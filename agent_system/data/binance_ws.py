"""Binance WebSocket market data engine for Phase 1."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import StreamDataError
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketDataBuffer, MarketTick
from agent_system.data.feature_engineering import FeatureEngine


logger = get_logger(__name__)


@dataclass
class StreamStats:
    messages_received: int = 0
    ticks_emitted: int = 0
    malformed_messages: int = 0
    reconnects: int = 0
    latency_ms_total: float = 0.0
    latency_samples: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.latency_samples == 0:
            return 0.0
        return self.latency_ms_total / self.latency_samples

    @property
    def packet_loss_estimate(self) -> int:
        return self.malformed_messages


class BinanceMarketStream:
    """Streams Binance aggTrade and depth data into a rolling tick buffer."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.symbol = config.symbol.lower()
        self.buffer = MarketDataBuffer(config.buffer_window_seconds)
        self.feature_engine = FeatureEngine()
        self.stats = StreamStats()
        self._best_bid: float | None = None
        self._best_ask: float | None = None
        self._stop_event = asyncio.Event()

    @property
    def stream_url(self) -> str:
        streams = f"{self.symbol}@aggTrade/{self.symbol}@depth5@100ms"
        return f"{self.config.binance_ws_base_url}?streams={streams}"

    def stop(self) -> None:
        self._stop_event.set()

    async def collect_ticks(self, min_ticks: int, timeout_seconds: float) -> list[MarketTick]:
        collector = asyncio.create_task(self.run())
        started = time.monotonic()
        try:
            while len(self.buffer) < min_ticks:
                if time.monotonic() - started >= timeout_seconds:
                    break
                await asyncio.sleep(0.1)
            return self.buffer.snapshot()
        finally:
            self.stop()
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)

    async def run(self) -> None:
        reconnect_delay = self.config.reconnect_initial_delay_seconds
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to Binance stream: %s", self.stream_url)
                async with websockets.connect(
                    self.stream_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                ) as websocket:
                    reconnect_delay = self.config.reconnect_initial_delay_seconds
                    await self._consume(websocket)
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, OSError, TimeoutError) as exc:
                self.stats.reconnects += 1
                logger.warning(
                    "Stream disconnected (%s). Reconnecting in %.1fs",
                    exc,
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 2,
                    self.config.reconnect_max_delay_seconds,
                )

    async def _consume(self, websocket: Any) -> None:
        while not self._stop_event.is_set():
            raw_message = await websocket.recv()
            self.stats.messages_received += 1
            try:
                await self._handle_message(raw_message)
            except StreamDataError as exc:
                self.stats.malformed_messages += 1
                logger.debug("Ignoring malformed stream message: %s", exc)

    async def _handle_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except JSONDecodeError as exc:
            raise StreamDataError("Invalid JSON") from exc

        stream = message.get("stream")
        payload = message.get("data")
        if not isinstance(stream, str) or not isinstance(payload, dict):
            raise StreamDataError("Missing stream or data payload")

        if stream.endswith("@depth5@100ms"):
            self._handle_depth(payload)
            return

        if stream.endswith("@aggTrade"):
            tick = self._handle_agg_trade(payload)
            if tick is not None:
                self.buffer.add(tick)
                self.stats.ticks_emitted += 1
                logger.info(
                    "%s live price %.2f | bid %.2f | ask %.2f | buffer %d ticks",
                    self.config.symbol,
                    tick.price,
                    tick.bid,
                    tick.ask,
                    len(self.buffer),
                )
            return

        raise StreamDataError(f"Unexpected stream: {stream}")

    def _handle_depth(self, payload: dict[str, Any]) -> None:
        bids = payload.get("bids")
        asks = payload.get("asks")
        if not bids or not asks:
            raise StreamDataError("Depth message missing bids or asks")

        self._best_bid = float(bids[0][0])
        self._best_ask = float(asks[0][0])

    def _handle_agg_trade(self, payload: dict[str, Any]) -> MarketTick | None:
        if self._best_bid is None or self._best_ask is None:
            return None

        event_time = int(payload["E"])
        current_ms = int(time.time() * 1000)
        latency_ms = max(0, current_ms - event_time)
        self.stats.latency_ms_total += latency_ms
        self.stats.latency_samples += 1

        return MarketTick(
            timestamp=event_time,
            price=float(payload["p"]),
            volume=float(payload["q"]),
            bid=self._best_bid,
            ask=self._best_ask,
        )


async def print_stream_stats(stream: BinanceMarketStream) -> None:
    while True:
        await asyncio.sleep(stream.config.stream_stats_interval_seconds)
        latest = stream.buffer.latest()
        latest_price = latest.price if latest else 0.0
        features = stream.feature_engine.compute(stream.buffer.snapshot())
        logger.info(
            "PHASE 1.5 stats | messages=%d ticks=%d reconnects=%d packet_loss_estimate=%d avg_latency_ms=%.1f buffer_ticks=%d latest_price=%.2f features=%s",
            stream.stats.messages_received,
            stream.stats.ticks_emitted,
            stream.stats.reconnects,
            stream.stats.packet_loss_estimate,
            stream.stats.avg_latency_ms,
            len(stream.buffer),
            latest_price,
            features.as_dict(),
        )


async def run_phase_one() -> None:
    stream = BinanceMarketStream()
    stats_task = asyncio.create_task(print_stream_stats(stream))
    try:
        await stream.run()
    finally:
        stats_task.cancel()
        await asyncio.gather(stats_task, return_exceptions=True)
