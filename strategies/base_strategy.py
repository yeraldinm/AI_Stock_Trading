"""
Base strategy framework for the low latency trading platform
"""
import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
import logging

import numpy as np
import pandas as pd

from core.models import (
    Order, Fill, Position, Portfolio, Signal, MarketData,
    OrderSide, OrderType, OrderStatus, TimeInForce, PerformanceMetrics
)

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, 
                 strategy_id: str,
                 name: str,
                 symbols: List[str],
                 parameters: Dict[str, Any] = None):
        self.strategy_id = strategy_id
        self.name = name
        self.symbols = symbols
        self.parameters = parameters or {}
        
        # Strategy state
        self.enabled = True
        self.positions = {}  # symbol -> Position
        self.orders = {}  # order_id -> Order
        self.signals = deque(maxlen=1000)
        self.market_data_buffer = {}  # symbol -> deque of MarketData
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.trade_history = deque(maxlen=10000)
        self.pnl_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.last_update = datetime.now(timezone.utc)
        
        # Callbacks
        self.signal_callbacks = []
        self.order_callbacks = []
        
        # Risk limits
        self.max_position_size = self.parameters.get('max_position_size', 10000.0)
        self.max_daily_loss = self.parameters.get('max_daily_loss', 1000.0)
        self.stop_loss_pct = self.parameters.get('stop_loss', 0.02)
        self.take_profit_pct = self.parameters.get('take_profit', 0.04)
        
        logger.info(f"Strategy initialized: {self.name} ({self.strategy_id})")
    
    @abstractmethod
    async def on_market_data(self, data: MarketData):
        """Handle incoming market data"""
        pass
    
    @abstractmethod
    async def on_signal(self, signal: Signal):
        """Handle trading signal"""
        pass
    
    @abstractmethod
    async def generate_signals(self) -> List[Signal]:
        """Generate trading signals based on current market conditions"""
        pass
    
    def add_signal_callback(self, callback: Callable[[Signal], None]):
        """Add callback for signal generation"""
        self.signal_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """Add callback for order placement"""
        self.order_callbacks.append(callback)
    
    async def update_market_data(self, data: MarketData):
        """Update market data buffer and trigger strategy logic"""
        try:
            # Update buffer
            if data.symbol not in self.market_data_buffer:
                self.market_data_buffer[data.symbol] = deque(maxlen=1000)
            self.market_data_buffer[data.symbol].append(data)
            
            # Update positions with current price
            if data.price and data.symbol in self.positions:
                self.positions[data.symbol].update_market_data(data.price)
            
            # Call strategy-specific handler
            await self.on_market_data(data)
            
            # Generate signals
            if self.enabled:
                signals = await self.generate_signals()
                for signal in signals:
                    await self._emit_signal(signal)
            
            self.last_update = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Market data update error in {self.name}: {e}")
    
    async def _emit_signal(self, signal: Signal):
        """Emit trading signal"""
        try:
            signal.strategy_id = self.strategy_id
            self.signals.append(signal)
            
            # Notify callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")
            
            # Handle signal internally
            await self.on_signal(signal)
            
        except Exception as e:
            logger.error(f"Signal emission error: {e}")
    
    def create_order(self, 
                    symbol: str,
                    side: OrderSide,
                    quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: TimeInForce = TimeInForce.DAY) -> Order:
        """Create order with strategy-specific settings"""
        
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_id=self.strategy_id
        )
        
        self.orders[order.id] = order
        return order
    
    async def place_order(self, order: Order):
        """Place order and notify callbacks"""
        try:
            # Validate order against strategy limits
            if not self._validate_order(order):
                logger.warning(f"Order validation failed: {order.id}")
                return
            
            # Notify callbacks
            for callback in self.order_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Order callback error: {e}")
            
            logger.info(f"Order placed by {self.name}: {order.symbol} {order.side.value} {order.quantity}")
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order against strategy limits"""
        try:
            # Check position size
            current_position = self.positions.get(order.symbol, Position(symbol=order.symbol))
            
            if order.is_buy:
                new_quantity = current_position.quantity + order.quantity
            else:
                new_quantity = current_position.quantity - order.quantity
            
            if abs(new_quantity) > self.max_position_size:
                logger.warning(f"Order exceeds position limit: {abs(new_quantity)} > {self.max_position_size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False
    
    def on_fill(self, fill: Fill):
        """Handle order fill"""
        try:
            # Update position
            if fill.symbol not in self.positions:
                self.positions[fill.symbol] = Position(symbol=fill.symbol)
            
            self.positions[fill.symbol].add_fill(fill)
            
            # Update performance metrics
            self._update_performance_metrics(fill)
            
            # Record trade
            self.trade_history.append({
                'timestamp': fill.timestamp,
                'symbol': fill.symbol,
                'side': fill.side,
                'quantity': fill.quantity,
                'price': fill.price,
                'pnl': self.positions[fill.symbol].realized_pnl
            })
            
            logger.info(f"Fill processed by {self.name}: {fill.symbol} {fill.quantity}@{fill.price}")
            
        except Exception as e:
            logger.error(f"Fill processing error: {e}")
    
    def _update_performance_metrics(self, fill: Fill):
        """Update strategy performance metrics"""
        try:
            position = self.positions[fill.symbol]
            
            # Update total trades
            self.performance_metrics.total_trades += 1
            
            # Check if this fill closed a position (realized P&L)
            if position.realized_pnl != 0:
                if position.realized_pnl > 0:
                    self.performance_metrics.winning_trades += 1
                    self.performance_metrics.average_win = (
                        (self.performance_metrics.average_win * (self.performance_metrics.winning_trades - 1) + 
                         position.realized_pnl) / self.performance_metrics.winning_trades
                    )
                    self.performance_metrics.largest_win = max(
                        self.performance_metrics.largest_win, position.realized_pnl
                    )
                else:
                    self.performance_metrics.losing_trades += 1
                    self.performance_metrics.average_loss = (
                        (self.performance_metrics.average_loss * (self.performance_metrics.losing_trades - 1) + 
                         abs(position.realized_pnl)) / self.performance_metrics.losing_trades
                    )
                    self.performance_metrics.largest_loss = max(
                        self.performance_metrics.largest_loss, abs(position.realized_pnl)
                    )
            
            # Update win rate
            if self.performance_metrics.total_trades > 0:
                self.performance_metrics.win_rate = (
                    self.performance_metrics.winning_trades / self.performance_metrics.total_trades
                )
            
            # Update profit factor
            if self.performance_metrics.average_loss > 0:
                total_wins = self.performance_metrics.winning_trades * self.performance_metrics.average_win
                total_losses = self.performance_metrics.losing_trades * self.performance_metrics.average_loss
                self.performance_metrics.profit_factor = total_wins / total_losses
            
            self.performance_metrics.last_update = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        if symbol in self.market_data_buffer and self.market_data_buffer[symbol]:
            return self.market_data_buffer[symbol][-1].price
        return None
    
    def get_price_history(self, symbol: str, count: int = 100) -> List[float]:
        """Get price history for symbol"""
        if symbol not in self.market_data_buffer:
            return []
        
        data = list(self.market_data_buffer[symbol])[-count:]
        return [d.price for d in data if d.price is not None]
    
    def calculate_returns(self, symbol: str, periods: int = 20) -> List[float]:
        """Calculate returns for symbol"""
        prices = self.get_price_history(symbol, periods + 1)
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        return returns
    
    def calculate_volatility(self, symbol: str, periods: int = 20) -> float:
        """Calculate volatility for symbol"""
        returns = self.calculate_returns(symbol, periods)
        if len(returns) < 2:
            return 0.0
        
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def calculate_moving_average(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Calculate moving average"""
        prices = self.get_price_history(symbol, periods)
        if len(prices) < periods:
            return None
        
        return sum(prices) / len(prices)
    
    def calculate_rsi(self, symbol: str, periods: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        returns = self.calculate_returns(symbol, periods + 1)
        if len(returns) < periods:
            return None
        
        gains = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        
        if not gains or not losses:
            return None
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_position(self, symbol: str) -> Position:
        """Get position for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def get_total_pnl(self) -> float:
        """Get total P&L across all positions"""
        total_pnl = 0.0
        for position in self.positions.values():
            total_pnl += position.unrealized_pnl + position.realized_pnl
        return total_pnl
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        elapsed = time.time() - self.start_time
        total_pnl = self.get_total_pnl()
        
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'enabled': self.enabled,
            'uptime_seconds': elapsed,
            'total_pnl': total_pnl,
            'total_trades': self.performance_metrics.total_trades,
            'win_rate': self.performance_metrics.win_rate,
            'profit_factor': self.performance_metrics.profit_factor,
            'sharpe_ratio': self.performance_metrics.sharpe_ratio,
            'max_drawdown': self.performance_metrics.max_drawdown,
            'active_positions': len([p for p in self.positions.values() if not p.is_flat]),
            'pending_orders': len([o for o in self.orders.values() if o.is_active]),
            'last_update': self.last_update.isoformat()
        }
    
    def enable(self):
        """Enable strategy"""
        self.enabled = True
        logger.info(f"Strategy enabled: {self.name}")
    
    def disable(self):
        """Disable strategy"""
        self.enabled = False
        logger.info(f"Strategy disabled: {self.name}")
    
    async def shutdown(self):
        """Shutdown strategy gracefully"""
        try:
            self.enabled = False
            
            # Cancel all pending orders
            active_orders = [o for o in self.orders.values() if o.is_active]
            for order in active_orders:
                order.status = OrderStatus.CANCELLED
            
            logger.info(f"Strategy shutdown: {self.name}")
            
        except Exception as e:
            logger.error(f"Strategy shutdown error: {e}")


class SignalBasedStrategy(BaseStrategy):
    """Base class for signal-based strategies"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_threshold = self.parameters.get('signal_threshold', 0.6)
        self.rebalance_frequency = self.parameters.get('rebalance_frequency', 300)  # seconds
        self.last_rebalance = time.time()
    
    async def on_signal(self, signal: Signal):
        """Handle trading signal"""
        try:
            if signal.strength < self.signal_threshold:
                return
            
            if signal.signal_type == "BUY":
                await self._handle_buy_signal(signal)
            elif signal.signal_type == "SELL":
                await self._handle_sell_signal(signal)
            
        except Exception as e:
            logger.error(f"Signal handling error: {e}")
    
    async def _handle_buy_signal(self, signal: Signal):
        """Handle buy signal"""
        try:
            current_position = self.get_position(signal.symbol)
            
            # Calculate order size based on signal strength and risk management
            order_size = min(
                signal.quantity,
                self.max_position_size - current_position.quantity
            )
            
            if order_size > 0:
                order = self.create_order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    quantity=order_size,
                    order_type=OrderType.MARKET
                )
                
                await self.place_order(order)
                
        except Exception as e:
            logger.error(f"Buy signal handling error: {e}")
    
    async def _handle_sell_signal(self, signal: Signal):
        """Handle sell signal"""
        try:
            current_position = self.get_position(signal.symbol)
            
            # Calculate order size
            order_size = min(
                signal.quantity,
                current_position.quantity
            )
            
            if order_size > 0:
                order = self.create_order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    quantity=order_size,
                    order_type=OrderType.MARKET
                )
                
                await self.place_order(order)
                
        except Exception as e:
            logger.error(f"Sell signal handling error: {e}")
    
    def should_rebalance(self) -> bool:
        """Check if strategy should rebalance"""
        return (time.time() - self.last_rebalance) >= self.rebalance_frequency
    
    async def rebalance_portfolio(self):
        """Rebalance portfolio based on current signals"""
        if not self.should_rebalance():
            return
        
        try:
            # Generate fresh signals
            signals = await self.generate_signals()
            
            # Process signals for rebalancing
            for signal in signals:
                await self.on_signal(signal)
            
            self.last_rebalance = time.time()
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing error: {e}")
    
    def get_performance_summary(self) -> dict:
        """Get strategy performance summary"""
        try:
            total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if getattr(t, 'pnl', 0) > 0])
            
            return {
                'strategy_id': self.strategy_id,
                'name': self.name,
                'enabled': self.enabled,
                'total_pnl': total_pnl,
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
                'realized_pnl': sum(pos.realized_pnl for pos in self.positions.values()),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'active_positions': len([p for p in self.positions.values() if not p.is_flat]),
                'active_orders': len([o for o in self.orders.values() if o.status in ['PENDING', 'PARTIALLY_FILLED']]),
                'uptime_seconds': time.time() - self.start_time,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'signals_generated': len(self.signals)
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'strategy_id': self.strategy_id,
                'name': self.name,
                'enabled': self.enabled,
                'error': str(e)
            }
    
    def get_position(self, symbol: str) -> 'Position':
        """Get position for symbol, creating empty one if doesn't exist"""
        if symbol not in self.positions:
            from core.models import Position
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def get_price_history(self, symbol: str, periods: int) -> List[float]:
        """Get price history for symbol"""
        if symbol not in self.market_data_buffer:
            return []
        
        prices = []
        for data in list(self.market_data_buffer[symbol])[-periods:]:
            if data.price:
                prices.append(data.price)
        
        return prices
    
    async def shutdown(self):
        """Shutdown strategy gracefully"""
        try:
            logger.info(f"Shutting down strategy {self.name} ({self.strategy_id})")
            
            # Disable strategy
            self.enabled = False
            
            # Cancel all pending orders
            for order in self.orders.values():
                if order.status in ['PENDING', 'PARTIALLY_FILLED']:
                    order.status = OrderStatus.CANCELLED
            
            # Log final performance
            performance = self.get_performance_summary()
            logger.info(f"Strategy {self.name} final performance: {performance}")
            
        except Exception as e:
            logger.error(f"Strategy shutdown error: {e}")