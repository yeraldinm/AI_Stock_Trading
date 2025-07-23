"""
Core data models for the Low Latency Trading Platform
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4, UUID

import numpy as np
import pandas as pd


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force enumeration"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class Tick:
    """Market tick data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: Optional[float] = None
    last_size: Optional[int] = None
    volume: Optional[int] = None
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2.0
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points"""
        return (self.spread / self.mid_price) * 10000


@dataclass
class OHLCV:
    """OHLCV bar data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str  # e.g., "1m", "5m", "1h", "1d"
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)"""
        return (self.high + self.low + self.close) / 3.0
    
    @property
    def range_pct(self) -> float:
        """Calculate price range as percentage"""
        return ((self.high - self.low) / self.open) * 100


@dataclass
class OrderBook:
    """Order book data structure"""
    symbol: str
    timestamp: datetime
    bids: List[tuple[float, int]]  # [(price, size), ...]
    asks: List[tuple[float, int]]  # [(price, size), ...]
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None


@dataclass
class Order:
    """Trading order"""
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity
    
    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order"""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order"""
        return self.side == OrderSide.SELL
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled)"""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    def update_fill(self, fill_quantity: float, fill_price: float):
        """Update order with fill information"""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average fill price
        total_filled_value = (self.average_fill_price * (self.filled_quantity - fill_quantity) + 
                             fill_price * fill_quantity)
        self.average_fill_price = total_filled_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now(timezone.utc)


@dataclass
class Fill:
    """Order fill/execution"""
    id: str = field(default_factory=lambda: str(uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    commission: float = 0.0
    exchange: Optional[str] = None
    execution_id: Optional[str] = None


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)"""
        return self.quantity == 0
    
    def update_market_data(self, current_price: float):
        """Update position with current market price"""
        if self.quantity != 0:
            self.market_value = self.quantity * current_price
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
        else:
            self.market_value = 0.0
            self.unrealized_pnl = 0.0
        self.last_update = datetime.now(timezone.utc)
    
    def add_fill(self, fill: Fill):
        """Add a fill to the position"""
        if fill.side == OrderSide.BUY:
            fill_quantity = fill.quantity
        else:
            fill_quantity = -fill.quantity
        
        # Calculate new average price
        if (self.quantity >= 0 and fill_quantity > 0) or (self.quantity <= 0 and fill_quantity < 0):
            # Same direction - update average price
            total_cost = self.quantity * self.average_price + fill_quantity * fill.price
            self.quantity += fill_quantity
            if self.quantity != 0:
                self.average_price = total_cost / self.quantity
        else:
            # Opposite direction - realize P&L
            if abs(fill_quantity) >= abs(self.quantity):
                # Close and potentially reverse position
                close_quantity = -self.quantity
                self.realized_pnl += close_quantity * (fill.price - self.average_price)
                
                remaining_quantity = fill_quantity + self.quantity
                if remaining_quantity != 0:
                    self.quantity = remaining_quantity
                    self.average_price = fill.price
                else:
                    self.quantity = 0
                    self.average_price = 0
            else:
                # Partial close
                self.realized_pnl += -fill_quantity * (fill.price - self.average_price)
                self.quantity += fill_quantity
        
        self.last_update = datetime.now(timezone.utc)


@dataclass
class Portfolio:
    """Trading portfolio"""
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 100000.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_position(self, symbol: str) -> Position:
        """Get position for symbol (create if doesn't exist)"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_market_data(self, symbol: str, price: float):
        """Update portfolio with market data"""
        if symbol in self.positions:
            self.positions[symbol].update_market_data(price)
        self._calculate_totals()
    
    def add_fill(self, fill: Fill):
        """Add fill to portfolio"""
        position = self.get_position(fill.symbol)
        position.add_fill(fill)
        
        # Update cash
        if fill.side == OrderSide.BUY:
            self.cash -= fill.quantity * fill.price + fill.commission
        else:
            self.cash += fill.quantity * fill.price - fill.commission
        
        self._calculate_totals()
    
    def _calculate_totals(self):
        """Calculate total portfolio values"""
        self.total_value = self.cash
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        for position in self.positions.values():
            self.total_value += position.market_value
            self.unrealized_pnl += position.unrealized_pnl
            self.realized_pnl += position.realized_pnl
        
        self.last_update = datetime.now(timezone.utc)


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio/strategy"""
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    volatility: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MarketData:
    """Consolidated market data"""
    symbol: str
    timestamp: datetime
    tick: Optional[Tick] = None
    ohlcv: Optional[OHLCV] = None
    order_book: Optional[OrderBook] = None
    
    @property
    def price(self) -> Optional[float]:
        """Get current price from available data"""
        if self.tick:
            return self.tick.last_price or self.tick.mid_price
        elif self.ohlcv:
            return self.ohlcv.close
        elif self.order_book:
            return self.order_book.mid_price
        return None


@dataclass
class Signal:
    """Trading signal"""
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    strategy_id: str = ""
    signal_type: str = ""  # "BUY", "SELL", "HOLD"
    strength: float = 0.0  # Signal strength 0-1
    price: float = 0.0
    quantity: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiry: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if signal is expired"""
        if self.expiry:
            return datetime.now(timezone.utc) > self.expiry
        return False


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))