"""
Main Low Latency Trading Platform
"""
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

import uvloop
import structlog

from config import config, DEFAULT_INSTRUMENTS, DEFAULT_STRATEGIES
from core.models import Portfolio, Order, Fill, MarketData
from core.market_data import MarketDataManager, WebSocketDataFeed
from core.order_management import OrderManager
from core.risk_management import RiskManager
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy

# Set up structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class TradingPlatform:
    """
    Main low latency trading platform that orchestrates all components
    """
    
    def __init__(self):
        self.portfolio = Portfolio(cash=100000.0)  # Start with $100k
        self.market_data_manager = None
        self.order_manager = None
        self.risk_manager = None
        self.strategies = {}
        
        # Platform state
        self.running = False
        self.start_time = None
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.stats = {
            'orders_placed': 0,
            'fills_processed': 0,
            'strategies_active': 0,
            'uptime_seconds': 0,
            'total_pnl': 0.0
        }
        
        logger.info("Trading platform initialized")
    
    async def initialize(self):
        """Initialize all platform components"""
        try:
            logger.info("Initializing trading platform components...")
            
            # Initialize market data manager
            self.market_data_manager = MarketDataManager()
            await self.market_data_manager.initialize()
            
            # Initialize order manager
            self.order_manager = OrderManager(self.portfolio)
            await self.order_manager.initialize()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.portfolio)
            await self.risk_manager.initialize()
            
            # Set up callbacks
            self._setup_callbacks()
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Set up market data feeds
            await self._setup_market_data_feeds()
            
            logger.info("Trading platform initialization completed")
            
        except Exception as e:
            logger.error("Platform initialization failed", error=str(e))
            raise
    
    def _setup_callbacks(self):
        """Set up callbacks between components"""
        # Market data callbacks
        self.market_data_manager.subscribe('*', self._on_market_data)
        
        # Order manager callbacks
        self.order_manager.fill_processor.add_callback(self._on_fill)
        
        # Risk manager callbacks
        self.risk_manager.risk_monitor.add_alert_callback(self._on_risk_alert)
    
    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            logger.info("Initializing trading strategies...")
            
            # Get symbols from instruments
            symbols = [inst.symbol for inst in DEFAULT_INSTRUMENTS if inst.enabled]
            
            # Initialize strategies based on configuration
            for strategy_config in DEFAULT_STRATEGIES:
                if not strategy_config.enabled:
                    continue
                
                strategy_id = f"{strategy_config.name}_{int(time.time())}"
                
                if strategy_config.name == "MeanReversion":
                    strategy = MeanReversionStrategy(
                        strategy_id=strategy_id,
                        symbols=symbols,
                        parameters=strategy_config.parameters
                    )
                else:
                    logger.warning(f"Unknown strategy type: {strategy_config.name}")
                    continue
                
                # Set up strategy callbacks
                strategy.add_order_callback(self._on_strategy_order)
                strategy.add_signal_callback(self._on_strategy_signal)
                
                self.strategies[strategy_id] = strategy
                
                logger.info(f"Initialized strategy: {strategy_config.name} ({strategy_id})")
            
            self.stats['strategies_active'] = len(self.strategies)
            logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error("Strategy initialization failed", error=str(e))
            raise
    
    async def _setup_market_data_feeds(self):
        """Set up market data feeds"""
        try:
            logger.info("Setting up market data feeds...")
            
            # Get symbols from instruments
            symbols = [inst.symbol for inst in DEFAULT_INSTRUMENTS if inst.enabled]
            
            # For demo purposes, we'll create a mock WebSocket feed
            # In production, this would connect to real market data providers
            alpaca_feed = WebSocketDataFeed(
                url="wss://stream.data.alpaca.markets/v2/iex",
                symbols=symbols
            )
            
            self.market_data_manager.add_feed("alpaca", alpaca_feed)
            
            logger.info(f"Set up market data feeds for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error("Market data feed setup failed", error=str(e))
            raise
    
    async def _on_market_data(self, data: MarketData):
        """Handle incoming market data"""
        try:
            # Update risk manager
            volume = None
            if data.tick and data.tick.volume:
                volume = data.tick.volume
            elif data.ohlcv:
                volume = data.ohlcv.volume
            
            if data.price:
                self.risk_manager.update_market_data(data.symbol, data.price, volume)
            
            # Update strategies
            for strategy in self.strategies.values():
                if data.symbol in strategy.symbols:
                    await strategy.update_market_data(data)
            
        except Exception as e:
            logger.error("Market data processing error", error=str(e))
    
    def _on_strategy_order(self, order: Order):
        """Handle order from strategy"""
        try:
            logger.info(f"Received order from strategy", 
                       strategy_id=order.strategy_id,
                       symbol=order.symbol,
                       side=order.side.value,
                       quantity=order.quantity)
            
            # Submit order asynchronously
            asyncio.create_task(self._process_strategy_order(order))
            
        except Exception as e:
            logger.error("Strategy order handling error", error=str(e))
    
    async def _process_strategy_order(self, order: Order):
        """Process order from strategy"""
        try:
            # Risk check
            risk_ok, risk_msg = self.risk_manager.check_order_risk(order)
            if not risk_ok:
                logger.warning(f"Order rejected by risk manager: {risk_msg}",
                              order_id=order.id)
                return
            
            # Submit order
            success, message = await self.order_manager.submit_order(order)
            
            if success:
                self.stats['orders_placed'] += 1
                logger.info(f"Order submitted successfully",
                           order_id=order.id,
                           message=message)
            else:
                logger.warning(f"Order submission failed",
                              order_id=order.id,
                              message=message)
            
        except Exception as e:
            logger.error("Order processing error", error=str(e))
    
    def _on_strategy_signal(self, signal):
        """Handle signal from strategy"""
        try:
            logger.info(f"Received signal from strategy",
                       strategy_id=signal.strategy_id,
                       symbol=signal.symbol,
                       signal_type=signal.signal_type,
                       strength=signal.strength)
            
        except Exception as e:
            logger.error("Strategy signal handling error", error=str(e))
    
    def _on_fill(self, fill: Fill):
        """Handle order fill"""
        try:
            self.stats['fills_processed'] += 1
            
            # Update strategy
            if fill.order_id in self.order_manager.orders:
                order = self.order_manager.orders[fill.order_id]
                if order.strategy_id in self.strategies:
                    strategy = self.strategies[order.strategy_id]
                    strategy.on_fill(fill)
            
            logger.info(f"Fill processed",
                       symbol=fill.symbol,
                       side=fill.side.value,
                       quantity=fill.quantity,
                       price=fill.price)
            
        except Exception as e:
            logger.error("Fill processing error", error=str(e))
    
    def _on_risk_alert(self, alert_type: str, message: str):
        """Handle risk alert"""
        try:
            logger.warning(f"Risk alert received",
                          alert_type=alert_type,
                          message=message)
            
            # Take action based on alert severity
            if alert_type in ['DAILY_LOSS_LIMIT', 'MAX_DRAWDOWN']:
                logger.critical("Severe risk alert - considering trading halt")
                # Could implement automatic trading halt here
            
        except Exception as e:
            logger.error("Risk alert handling error", error=str(e))
    
    async def start(self):
        """Start the trading platform"""
        try:
            if self.running:
                logger.warning("Platform is already running")
                return
            
            logger.info("Starting trading platform...")
            self.running = True
            self.start_time = time.time()
            
            # Start all components
            tasks = []
            
            # Start market data manager
            market_data_task = asyncio.create_task(
                self.market_data_manager.start()
            )
            tasks.append(market_data_task)
            
            # Start order monitoring
            order_monitoring_task = asyncio.create_task(
                self.order_manager.start_order_monitoring()
            )
            tasks.append(order_monitoring_task)
            
            # Start performance monitoring
            perf_task = asyncio.create_task(self._performance_monitor())
            tasks.append(perf_task)
            
            # Start health check
            health_task = asyncio.create_task(self._health_monitor())
            tasks.append(health_task)
            
            logger.info("Trading platform started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error("Platform start error", error=str(e))
            raise
        finally:
            self.running = False
    
    async def _performance_monitor(self):
        """Monitor platform performance"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Update stats
                self.stats['uptime_seconds'] = time.time() - self.start_time
                self.stats['total_pnl'] = self.portfolio.unrealized_pnl + self.portfolio.realized_pnl
                
                # Log performance metrics
                market_data_stats = self.market_data_manager.get_stats()
                order_stats = self.order_manager.get_stats()
                risk_stats = self.risk_manager.get_risk_summary()
                
                logger.info("Platform performance update",
                           platform_stats=self.stats,
                           market_data_stats=market_data_stats,
                           order_stats=order_stats,
                           risk_stats=risk_stats)
                
            except Exception as e:
                logger.error("Performance monitoring error", error=str(e))
                await asyncio.sleep(60)
    
    async def _health_monitor(self):
        """Monitor platform health"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check component health
                health_status = {
                    'market_data_manager': self.market_data_manager.running,
                    'order_manager': len(self.order_manager.active_orders),
                    'risk_manager': self.risk_manager.trading_enabled,
                    'strategies': {
                        sid: strategy.enabled 
                        for sid, strategy in self.strategies.items()
                    }
                }
                
                logger.info("Platform health check", health_status=health_status)
                
                # Check for issues
                if not self.market_data_manager.running:
                    logger.error("Market data manager is not running")
                
                if self.risk_manager.emergency_stop:
                    logger.critical("Emergency stop is active")
                
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def stop(self):
        """Stop the trading platform gracefully"""
        try:
            logger.info("Stopping trading platform...")
            
            # Stop strategies
            for strategy in self.strategies.values():
                await strategy.shutdown()
            
            # Cancel all orders
            await self.order_manager.cancel_all_orders()
            
            # Stop market data feeds
            await self.market_data_manager.stop()
            
            # Signal shutdown
            self.shutdown_event.set()
            
            logger.info("Trading platform stopped")
            
        except Exception as e:
            logger.error("Platform stop error", error=str(e))
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status"""
        return {
            'running': self.running,
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
            'stats': self.stats,
            'portfolio': {
                'total_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'unrealized_pnl': self.portfolio.unrealized_pnl,
                'realized_pnl': self.portfolio.realized_pnl,
                'positions': len([p for p in self.portfolio.positions.values() if not p.is_flat])
            },
            'strategies': {
                sid: strategy.get_performance_summary()
                for sid, strategy in self.strategies.items()
            },
            'risk_metrics': self.risk_manager.get_risk_summary() if self.risk_manager else {},
            'market_data': self.market_data_manager.get_stats() if self.market_data_manager else {},
            'orders': self.order_manager.get_stats() if self.order_manager else {}
        }


async def main():
    """Main entry point"""
    # Set up uvloop for better performance
    if sys.platform != 'win32':
        uvloop.install()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create platform
    platform = TradingPlatform()
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(platform.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and start platform
        await platform.initialize()
        await platform.start()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error("Platform error", error=str(e))
    finally:
        await platform.stop()


if __name__ == "__main__":
    asyncio.run(main())