"""
Mean Reversion Trading Strategy
"""
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
import logging

import numpy as np

from core.models import Signal, MarketData, OrderSide
from .base_strategy import SignalBasedStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(SignalBasedStrategy):
    """
    Mean reversion strategy that trades when price deviates significantly
    from its moving average, expecting it to revert to the mean.
    """
    
    def __init__(self, strategy_id: str, symbols: List[str], parameters: dict = None):
        default_params = {
            'lookback_period': 20,
            'deviation_threshold': 2.0,  # Standard deviations
            'stop_loss': 0.02,  # 2%
            'take_profit': 0.01,  # 1%
            'min_volume': 100000,  # Minimum daily volume
            'max_position_size': 1000.0,
            'signal_threshold': 0.7
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            strategy_id=strategy_id,
            name="MeanReversion",
            symbols=symbols,
            parameters=default_params
        )
        
        self.lookback_period = self.parameters['lookback_period']
        self.deviation_threshold = self.parameters['deviation_threshold']
        self.min_volume = self.parameters['min_volume']
        
        logger.info(f"Mean Reversion Strategy initialized with parameters: {self.parameters}")
    
    async def on_market_data(self, data: MarketData):
        """Process incoming market data"""
        try:
            if not data.price:
                return
            
            # Update volume tracking if available
            if data.tick and data.tick.volume:
                self._update_volume_tracking(data.symbol, data.tick.volume)
            
            # Check if we have enough data for analysis
            if not self._has_sufficient_data(data.symbol):
                return
            
            # Calculate mean reversion signals
            await self._analyze_mean_reversion(data.symbol, data.price)
            
        except Exception as e:
            logger.error(f"Market data processing error: {e}")
    
    def _update_volume_tracking(self, symbol: str, volume: int):
        """Update volume tracking for liquidity checks"""
        if not hasattr(self, 'volume_history'):
            self.volume_history = {}
        
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        
        self.volume_history[symbol].append(volume)
        
        # Keep only recent volume data
        if len(self.volume_history[symbol]) > 100:
            self.volume_history[symbol] = self.volume_history[symbol][-100:]
    
    def _has_sufficient_data(self, symbol: str) -> bool:
        """Check if we have sufficient data for analysis"""
        if symbol not in self.market_data_buffer:
            return False
        
        return len(self.market_data_buffer[symbol]) >= self.lookback_period
    
    async def _analyze_mean_reversion(self, symbol: str, current_price: float):
        """Analyze mean reversion opportunity"""
        try:
            # Calculate moving average and standard deviation
            prices = self.get_price_history(symbol, self.lookback_period)
            if len(prices) < self.lookback_period:
                return
            
            moving_average = np.mean(prices)
            std_dev = np.std(prices)
            
            if std_dev == 0:
                return
            
            # Calculate z-score (number of standard deviations from mean)
            z_score = (current_price - moving_average) / std_dev
            
            # Check for mean reversion signals
            signal_strength = min(abs(z_score) / self.deviation_threshold, 1.0)
            
            if abs(z_score) >= self.deviation_threshold:
                # Price has deviated significantly - expect reversion
                if z_score > 0:
                    # Price is above mean - sell signal
                    signal_type = "SELL"
                    target_price = moving_average
                else:
                    # Price is below mean - buy signal
                    signal_type = "BUY"
                    target_price = moving_average
                
                # Calculate position size based on volatility and signal strength
                position_size = self._calculate_position_size(
                    symbol, current_price, std_dev, signal_strength
                )
                
                if position_size > 0:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=signal_strength,
                        price=current_price,
                        quantity=position_size,
                        metadata={
                            'z_score': z_score,
                            'moving_average': moving_average,
                            'std_dev': std_dev,
                            'target_price': target_price,
                            'strategy': 'mean_reversion'
                        }
                    )
                    
                    await self._emit_signal(signal)
            
        except Exception as e:
            logger.error(f"Mean reversion analysis error: {e}")
    
    def _calculate_position_size(self, symbol: str, price: float, volatility: float, signal_strength: float) -> float:
        """Calculate position size based on volatility and signal strength"""
        try:
            # Base position size
            base_size = self.max_position_size * 0.1  # 10% of max position
            
            # Adjust for signal strength
            size_adjustment = signal_strength
            
            # Adjust for volatility (reduce size for higher volatility)
            volatility_adjustment = max(0.5, 1.0 - (volatility * 10))
            
            # Check current position
            current_position = self.get_position(symbol)
            
            # Calculate final position size
            position_size = base_size * size_adjustment * volatility_adjustment
            
            # Ensure we don't exceed maximum position
            max_additional = self.max_position_size - abs(current_position.quantity)
            position_size = min(position_size, max_additional)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0
    
    async def generate_signals(self) -> List[Signal]:
        """Generate mean reversion signals for all symbols"""
        signals = []
        
        try:
            for symbol in self.symbols:
                if not self._has_sufficient_data(symbol):
                    continue
                
                current_price = self.get_latest_price(symbol)
                if not current_price:
                    continue
                
                # Check liquidity
                if not self._check_liquidity(symbol):
                    continue
                
                # Analyze for mean reversion
                await self._analyze_mean_reversion(symbol, current_price)
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
        
        return signals
    
    def _check_liquidity(self, symbol: str) -> bool:
        """Check if symbol has sufficient liquidity"""
        if not hasattr(self, 'volume_history') or symbol not in self.volume_history:
            return True  # Assume sufficient liquidity if no data
        
        recent_volume = self.volume_history[symbol][-10:]  # Last 10 periods
        avg_volume = np.mean(recent_volume) if recent_volume else 0
        
        return avg_volume >= self.min_volume
    
    def _should_close_position(self, symbol: str, current_price: float) -> bool:
        """Check if position should be closed based on stop loss or take profit"""
        position = self.get_position(symbol)
        
        if position.is_flat:
            return False
        
        # Calculate P&L percentage
        pnl_pct = (current_price - position.average_price) / position.average_price
        
        if position.is_long:
            # Long position
            if pnl_pct <= -self.stop_loss_pct:
                logger.info(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                return True
            elif pnl_pct >= self.take_profit_pct:
                logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%}")
                return True
        else:
            # Short position
            if pnl_pct >= self.stop_loss_pct:
                logger.info(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                return True
            elif pnl_pct <= -self.take_profit_pct:
                logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%}")
                return True
        
        return False
    
    async def check_exit_conditions(self):
        """Check exit conditions for all positions"""
        try:
            for symbol in self.symbols:
                position = self.get_position(symbol)
                if position.is_flat:
                    continue
                
                current_price = self.get_latest_price(symbol)
                if not current_price:
                    continue
                
                if self._should_close_position(symbol, current_price):
                    # Close position
                    side = OrderSide.SELL if position.is_long else OrderSide.BUY
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type="SELL" if position.is_long else "BUY",
                        strength=1.0,  # High strength for exit
                        price=current_price,
                        quantity=abs(position.quantity),
                        metadata={
                            'reason': 'exit_condition',
                            'strategy': 'mean_reversion'
                        }
                    )
                    
                    await self._emit_signal(signal)
            
        except Exception as e:
            logger.error(f"Exit condition check error: {e}")
    
    async def on_signal(self, signal: Signal):
        """Handle trading signal with additional mean reversion logic"""
        try:
            # First check exit conditions
            await self.check_exit_conditions()
            
            # Then handle the signal
            await super().on_signal(signal)
            
        except Exception as e:
            logger.error(f"Signal handling error: {e}")
    
    def get_strategy_state(self) -> dict:
        """Get current strategy state for monitoring"""
        state = {
            'name': self.name,
            'enabled': self.enabled,
            'parameters': self.parameters,
            'positions': {},
            'recent_signals': []
        }
        
        # Add position information
        for symbol, position in self.positions.items():
            if not position.is_flat:
                current_price = self.get_latest_price(symbol)
                pnl_pct = 0
                if current_price and position.average_price:
                    pnl_pct = (current_price - position.average_price) / position.average_price
                
                state['positions'][symbol] = {
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'current_price': current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'pnl_pct': pnl_pct
                }
        
        # Add recent signals
        recent_signals = list(self.signals)[-5:]  # Last 5 signals
        for signal in recent_signals:
            state['recent_signals'].append({
                'symbol': signal.symbol,
                'type': signal.signal_type,
                'strength': signal.strength,
                'timestamp': signal.timestamp.isoformat(),
                'metadata': signal.metadata
            })
        
        return state