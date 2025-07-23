"""
Comprehensive risk management system for low latency trading
"""
import asyncio
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
import logging
import math

import numpy as np
import pandas as pd
import redis
from kafka import KafkaProducer

from .models import (
    Order, Fill, Position, Portfolio, OrderSide, OrderType, 
    OrderStatus, RiskMetrics, PerformanceMetrics
)
from config import config

logger = logging.getLogger(__name__)


class PositionRiskManager:
    """Position-level risk management"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.position_limits = {}  # symbol -> max_position
        self.concentration_limit = 0.20  # Max 20% of portfolio in single position
        self.sector_limits = {}  # sector -> max_exposure
        
    def set_position_limit(self, symbol: str, max_position: float):
        """Set position limit for symbol"""
        self.position_limits[symbol] = max_position
    
    def set_concentration_limit(self, limit: float):
        """Set concentration limit (0-1)"""
        self.concentration_limit = limit
    
    def check_position_risk(self, order: Order) -> tuple[bool, str]:
        """Check if order violates position risk limits"""
        try:
            current_position = self.portfolio.get_position(order.symbol)
            
            # Calculate new position after order
            if order.is_buy:
                new_quantity = current_position.quantity + order.quantity
            else:
                new_quantity = current_position.quantity - order.quantity
            
            # Check position limit
            symbol_limit = self.position_limits.get(order.symbol, config.MAX_POSITION_SIZE)
            if abs(new_quantity) > symbol_limit:
                return False, f"Position limit exceeded: {abs(new_quantity)} > {symbol_limit}"
            
            # Check concentration limit
            if order.price:
                new_position_value = abs(new_quantity * order.price)
                concentration = new_position_value / self.portfolio.total_value
                if concentration > self.concentration_limit:
                    return False, f"Concentration limit exceeded: {concentration:.2%} > {self.concentration_limit:.2%}"
            
            return True, "Position risk check passed"
            
        except Exception as e:
            logger.error(f"Position risk check error: {e}")
            return False, f"Risk check error: {e}"


class PnLRiskManager:
    """P&L-based risk management"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.daily_loss_limit = config.MAX_DAILY_LOSS
        self.max_drawdown_limit = 0.10  # 10% max drawdown
        self.var_limit = 0.05  # 5% VaR limit
        
        # Track daily P&L
        self.daily_pnl = 0.0
        self.daily_start_value = portfolio.total_value
        self.last_reset = datetime.now(timezone.utc).date()
        
        # Track drawdown
        self.peak_value = portfolio.total_value
        self.max_drawdown = 0.0
        
    def update_daily_pnl(self):
        """Update daily P&L tracking"""
        today = datetime.now(timezone.utc).date()
        if today > self.last_reset:
            # New day - reset counters
            self.daily_start_value = self.portfolio.total_value
            self.daily_pnl = 0.0
            self.last_reset = today
        else:
            # Update daily P&L
            self.daily_pnl = self.portfolio.total_value - self.daily_start_value
        
        # Update drawdown
        if self.portfolio.total_value > self.peak_value:
            self.peak_value = self.portfolio.total_value
        
        current_drawdown = (self.peak_value - self.portfolio.total_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def check_pnl_risk(self, order: Order) -> tuple[bool, str]:
        """Check if order violates P&L risk limits"""
        try:
            self.update_daily_pnl()
            
            # Check daily loss limit
            if self.daily_pnl < -self.daily_loss_limit:
                return False, f"Daily loss limit exceeded: ${self.daily_pnl:.2f} < ${-self.daily_loss_limit:.2f}"
            
            # Check drawdown limit
            if self.max_drawdown > self.max_drawdown_limit:
                return False, f"Max drawdown exceeded: {self.max_drawdown:.2%} > {self.max_drawdown_limit:.2%}"
            
            return True, "P&L risk check passed"
            
        except Exception as e:
            logger.error(f"P&L risk check error: {e}")
            return False, f"P&L risk check error: {e}"


class VolatilityRiskManager:
    """Volatility-based risk management"""
    
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volatility_cache = {}
        self.volatility_limits = {}  # symbol -> max_volatility
        self.correlation_matrix = {}
        
    def update_price(self, symbol: str, price: float):
        """Update price history for volatility calculation"""
        self.price_history[symbol].append({
            'price': price,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Recalculate volatility if we have enough data
        if len(self.price_history[symbol]) >= 20:
            self._calculate_volatility(symbol)
    
    def _calculate_volatility(self, symbol: str):
        """Calculate historical volatility"""
        try:
            prices = [p['price'] for p in self.price_history[symbol]]
            if len(prices) < 2:
                return
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            # Calculate volatility (annualized)
            if returns:
                vol = np.std(returns) * np.sqrt(252)  # Annualized
                self.volatility_cache[symbol] = vol
                
        except Exception as e:
            logger.error(f"Volatility calculation error for {symbol}: {e}")
    
    def get_volatility(self, symbol: str) -> Optional[float]:
        """Get volatility for symbol"""
        return self.volatility_cache.get(symbol)
    
    def set_volatility_limit(self, symbol: str, max_vol: float):
        """Set volatility limit for symbol"""
        self.volatility_limits[symbol] = max_vol
    
    def check_volatility_risk(self, order: Order) -> tuple[bool, str]:
        """Check if order violates volatility limits"""
        try:
            vol = self.get_volatility(order.symbol)
            if vol is None:
                return True, "No volatility data available"
            
            vol_limit = self.volatility_limits.get(order.symbol, 1.0)  # Default 100% vol limit
            if vol > vol_limit:
                return False, f"Volatility limit exceeded: {vol:.2%} > {vol_limit:.2%}"
            
            return True, "Volatility risk check passed"
            
        except Exception as e:
            logger.error(f"Volatility risk check error: {e}")
            return False, f"Volatility risk check error: {e}"


class LiquidityRiskManager:
    """Liquidity-based risk management"""
    
    def __init__(self):
        self.volume_history = defaultdict(lambda: deque(maxlen=50))
        self.avg_volume = {}
        self.liquidity_limits = {}  # symbol -> max_percentage_of_volume
        
    def update_volume(self, symbol: str, volume: int):
        """Update volume history"""
        self.volume_history[symbol].append(volume)
        
        # Calculate average volume
        if self.volume_history[symbol]:
            self.avg_volume[symbol] = sum(self.volume_history[symbol]) / len(self.volume_history[symbol])
    
    def set_liquidity_limit(self, symbol: str, max_percentage: float):
        """Set liquidity limit as percentage of average volume"""
        self.liquidity_limits[symbol] = max_percentage
    
    def check_liquidity_risk(self, order: Order) -> tuple[bool, str]:
        """Check if order violates liquidity limits"""
        try:
            avg_vol = self.avg_volume.get(order.symbol)
            if avg_vol is None:
                return True, "No volume data available"
            
            liquidity_limit = self.liquidity_limits.get(order.symbol, 0.10)  # Default 10% of volume
            max_quantity = avg_vol * liquidity_limit
            
            if order.quantity > max_quantity:
                return False, f"Liquidity limit exceeded: {order.quantity} > {max_quantity:.0f} (10% of avg volume)"
            
            return True, "Liquidity risk check passed"
            
        except Exception as e:
            logger.error(f"Liquidity risk check error: {e}")
            return False, f"Liquidity risk check error: {e}"


class RiskMetricsCalculator:
    """Calculate comprehensive risk metrics"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.returns_history = deque(maxlen=252)  # 1 year of daily returns
        self.portfolio_values = deque(maxlen=252)
        
    def update_portfolio_value(self, value: float):
        """Update portfolio value for risk calculations"""
        self.portfolio_values.append({
            'value': value,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Calculate return if we have previous value
        if len(self.portfolio_values) >= 2:
            prev_value = self.portfolio_values[-2]['value']
            current_value = self.portfolio_values[-1]['value']
            ret = (current_value - prev_value) / prev_value
            self.returns_history.append(ret)
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = list(self.returns_history)
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * len(sorted_returns))
        var = sorted_returns[var_index] * self.portfolio.total_value
        return abs(var)
    
    def calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = list(self.returns_history)
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * len(sorted_returns))
        tail_returns = sorted_returns[:var_index]
        
        if tail_returns:
            es = np.mean(tail_returns) * self.portfolio.total_value
            return abs(es)
        return 0.0
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = list(self.returns_history)
        avg_return = np.mean(returns) * 252  # Annualized
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        if volatility == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / volatility
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        values = [pv['value'] for pv in self.portfolio_values]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        return RiskMetrics(
            var_95=self.calculate_var(0.95),
            var_99=self.calculate_var(0.99),
            expected_shortfall=self.calculate_expected_shortfall(),
            max_drawdown=self.calculate_max_drawdown(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            volatility=np.std(list(self.returns_history)) * np.sqrt(252) if self.returns_history else 0.0,
            last_update=datetime.now(timezone.utc)
        )


class RiskMonitor:
    """Real-time risk monitoring and alerting"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.risk_alerts = []
        self.alert_callbacks = []
        self.monitoring_active = False
        
        # Risk thresholds
        self.thresholds = {
            'daily_loss': config.MAX_DAILY_LOSS,
            'max_drawdown': 0.10,
            'var_95': 0.05,
            'concentration': 0.20,
            'volatility': 0.50
        }
        
    def add_alert_callback(self, callback: Callable[[str, str], None]):
        """Add callback for risk alerts"""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric: str, value: float):
        """Set risk threshold"""
        self.thresholds[metric] = value
    
    async def start_monitoring(self):
        """Start real-time risk monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self._check_risk_levels()
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _check_risk_levels(self):
        """Check current risk levels against thresholds"""
        try:
            # Check portfolio-level risks
            total_value = self.portfolio.total_value
            unrealized_pnl = self.portfolio.unrealized_pnl
            
            # Daily P&L check
            daily_pnl = unrealized_pnl  # Simplified - should track from day start
            if daily_pnl < -self.thresholds['daily_loss']:
                await self._trigger_alert(
                    'DAILY_LOSS_LIMIT',
                    f"Daily loss limit breached: ${daily_pnl:.2f}"
                )
            
            # Position concentration check
            for symbol, position in self.portfolio.positions.items():
                if position.market_value > 0:
                    concentration = position.market_value / total_value
                    if concentration > self.thresholds['concentration']:
                        await self._trigger_alert(
                            'CONCENTRATION_LIMIT',
                            f"Concentration limit breached for {symbol}: {concentration:.2%}"
                        )
            
        except Exception as e:
            logger.error(f"Risk level check error: {e}")
    
    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger risk alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(timezone.utc),
            'portfolio_value': self.portfolio.total_value
        }
        
        self.risk_alerts.append(alert)
        logger.warning(f"RISK ALERT: {alert_type} - {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent risk alerts"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [alert for alert in self.risk_alerts 
                if alert['timestamp'] > cutoff]


class RiskManager:
    """Central risk management system"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        
        # Risk managers
        self.position_risk = PositionRiskManager(portfolio)
        self.pnl_risk = PnLRiskManager(portfolio)
        self.volatility_risk = VolatilityRiskManager()
        self.liquidity_risk = LiquidityRiskManager()
        
        # Risk calculator and monitor
        self.risk_calculator = RiskMetricsCalculator(portfolio)
        self.risk_monitor = RiskMonitor(portfolio)
        
        # Risk state
        self.trading_enabled = True
        self.emergency_stop = False
        self.risk_checks_enabled = True
        
        # Performance tracking
        self.risk_check_times = deque(maxlen=1000)
        
    async def initialize(self):
        """Initialize risk management system"""
        try:
            # Start risk monitoring
            asyncio.create_task(self.risk_monitor.start_monitoring())
            
            # Set up alert callbacks
            self.risk_monitor.add_alert_callback(self._on_risk_alert)
            
            logger.info("Risk management system initialized")
            
        except Exception as e:
            logger.error(f"Risk manager initialization error: {e}")
            raise
    
    def check_order_risk(self, order: Order) -> tuple[bool, str]:
        """Comprehensive order risk check"""
        if not self.risk_checks_enabled:
            return True, "Risk checks disabled"
        
        if self.emergency_stop:
            return False, "Emergency stop activated"
        
        if not self.trading_enabled:
            return False, "Trading disabled"
        
        start_time = time.time()
        
        try:
            # Position risk check
            pos_ok, pos_msg = self.position_risk.check_position_risk(order)
            if not pos_ok:
                return False, f"Position risk: {pos_msg}"
            
            # P&L risk check
            pnl_ok, pnl_msg = self.pnl_risk.check_pnl_risk(order)
            if not pnl_ok:
                return False, f"P&L risk: {pnl_msg}"
            
            # Volatility risk check
            vol_ok, vol_msg = self.volatility_risk.check_volatility_risk(order)
            if not vol_ok:
                return False, f"Volatility risk: {vol_msg}"
            
            # Liquidity risk check
            liq_ok, liq_msg = self.liquidity_risk.check_liquidity_risk(order)
            if not liq_ok:
                return False, f"Liquidity risk: {liq_msg}"
            
            return True, "All risk checks passed"
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False, f"Risk check error: {e}"
        
        finally:
            # Record check time
            check_time = (time.time() - start_time) * 1000  # ms
            self.risk_check_times.append(check_time)
    
    def update_market_data(self, symbol: str, price: float, volume: Optional[int] = None):
        """Update risk managers with market data"""
        try:
            self.volatility_risk.update_price(symbol, price)
            
            if volume:
                self.liquidity_risk.update_volume(symbol, volume)
            
            # Update portfolio for risk calculations
            self.portfolio.update_market_data(symbol, price)
            self.risk_calculator.update_portfolio_value(self.portfolio.total_value)
            
        except Exception as e:
            logger.error(f"Market data update error: {e}")
    
    def _on_risk_alert(self, alert_type: str, message: str):
        """Handle risk alerts"""
        logger.warning(f"Risk alert received: {alert_type} - {message}")
        
        # Take action based on alert type
        if alert_type in ['DAILY_LOSS_LIMIT', 'MAX_DRAWDOWN']:
            # Severe alerts - consider stopping trading
            logger.critical(f"Severe risk alert: {alert_type}")
            # Could implement automatic trading halt here
    
    def enable_trading(self):
        """Enable trading"""
        self.trading_enabled = True
        logger.info("Trading enabled")
    
    def disable_trading(self):
        """Disable trading"""
        self.trading_enabled = False
        logger.warning("Trading disabled")
    
    def emergency_stop_all(self):
        """Emergency stop - halt all trading immediately"""
        self.emergency_stop = True
        self.trading_enabled = False
        logger.critical("EMERGENCY STOP ACTIVATED")
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop = False
        logger.info("Emergency stop reset")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        return self.risk_calculator.get_risk_metrics()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        risk_metrics = self.get_risk_metrics()
        
        return {
            'trading_enabled': self.trading_enabled,
            'emergency_stop': self.emergency_stop,
            'risk_checks_enabled': self.risk_checks_enabled,
            'daily_pnl': self.pnl_risk.daily_pnl,
            'max_drawdown': risk_metrics.max_drawdown,
            'var_95': risk_metrics.var_95,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'recent_alerts': len(self.risk_monitor.get_recent_alerts()),
            'avg_risk_check_time_ms': (
                sum(self.risk_check_times) / len(self.risk_check_times) 
                if self.risk_check_times else 0
            )
        }
    
    def set_position_limit(self, symbol: str, limit: float):
        """Set position limit for symbol"""
        self.position_risk.set_position_limit(symbol, limit)
    
    def set_volatility_limit(self, symbol: str, limit: float):
        """Set volatility limit for symbol"""
        self.volatility_risk.set_volatility_limit(symbol, limit)
    
    def set_liquidity_limit(self, symbol: str, limit: float):
        """Set liquidity limit for symbol"""
        self.liquidity_risk.set_liquidity_limit(symbol, limit)