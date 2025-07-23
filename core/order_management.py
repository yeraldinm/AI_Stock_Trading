"""
High-performance order management system for low latency trading
"""
import asyncio
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any, Set
import logging
from uuid import uuid4

import redis
import alpaca_trade_api as tradeapi
from kafka import KafkaProducer
import json

from .models import (
    Order, Fill, Position, Portfolio, OrderSide, OrderType, 
    OrderStatus, TimeInForce, MarketData
)
from config import config

logger = logging.getLogger(__name__)


class OrderValidator:
    """Order validation and pre-trade risk checks"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.max_order_size = config.MAX_POSITION_SIZE
        self.max_orders_per_second = config.MAX_ORDERS_PER_SECOND
        self.order_timestamps = deque(maxlen=100)
    
    def validate_order(self, order: Order) -> tuple[bool, str]:
        """Validate order before submission"""
        try:
            # Check order size
            if order.quantity <= 0:
                return False, "Order quantity must be positive"
            
            if order.quantity > self.max_order_size:
                return False, f"Order size exceeds maximum: {self.max_order_size}"
            
            # Check rate limiting
            now = time.time()
            recent_orders = [ts for ts in self.order_timestamps if now - ts < 1.0]
            if len(recent_orders) >= self.max_orders_per_second:
                return False, f"Order rate limit exceeded: {self.max_orders_per_second}/sec"
            
            # Check buying power
            if order.is_buy:
                required_cash = order.quantity * (order.price or 0)
                if required_cash > self.portfolio.cash:
                    return False, f"Insufficient buying power: {self.portfolio.cash}"
            
            # Check position limits
            current_position = self.portfolio.get_position(order.symbol)
            if order.is_buy:
                new_position = current_position.quantity + order.quantity
            else:
                new_position = current_position.quantity - order.quantity
            
            if abs(new_position) > self.max_order_size:
                return False, f"Position would exceed limit: {self.max_order_size}"
            
            # Record order timestamp for rate limiting
            self.order_timestamps.append(now)
            
            return True, "Order validated"
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False, f"Validation error: {e}"


class OrderRouter:
    """Smart order routing to optimize execution"""
    
    def __init__(self):
        self.venues = {}
        self.routing_rules = {}
    
    def add_venue(self, name: str, venue):
        """Add trading venue"""
        self.venues[name] = venue
    
    def route_order(self, order: Order) -> str:
        """Route order to best venue"""
        # Simple routing logic - can be enhanced with:
        # - Venue connectivity
        # - Historical fill rates
        # - Market impact analysis
        # - Latency considerations
        
        if order.symbol in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
            return "alpaca"  # Route to Alpaca for major stocks
        
        return "alpaca"  # Default routing


class ExecutionEngine:
    """Low latency order execution engine"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.pending_orders = {}  # order_id -> Order
        self.order_callbacks = defaultdict(list)
        self.fill_callbacks = []
        self.execution_times = deque(maxlen=1000)
        self.lock = threading.RLock()
    
    def add_order_callback(self, order_id: str, callback: Callable[[Order], None]):
        """Add callback for order updates"""
        self.order_callbacks[order_id].append(callback)
    
    def add_fill_callback(self, callback: Callable[[Fill], None]):
        """Add callback for fill updates"""
        self.fill_callbacks.append(callback)
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order for execution"""
        start_time = time.time()
        
        try:
            with self.lock:
                self.pending_orders[order.id] = order
            
            # Convert to broker format
            broker_order = self._convert_to_broker_format(order)
            
            # Submit to broker
            response = await self._submit_to_broker(broker_order)
            
            if response:
                order.broker_order_id = response.get('id')
                order.status = OrderStatus.SUBMITTED
                order.updated_at = datetime.now(timezone.utc)
                
                # Record execution time
                execution_time = (time.time() - start_time) * 1000  # ms
                self.execution_times.append(execution_time)
                
                logger.info(f"Order submitted: {order.id} in {execution_time:.2f}ms")
                
                # Notify callbacks
                await self._notify_order_callbacks(order)
                
                return True
            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now(timezone.utc)
                await self._notify_order_callbacks(order)
                return False
                
        except Exception as e:
            logger.error(f"Order submission error: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now(timezone.utc)
            await self._notify_order_callbacks(order)
            return False
    
    def _convert_to_broker_format(self, order: Order) -> Dict[str, Any]:
        """Convert internal order to broker format"""
        broker_order = {
            'symbol': order.symbol,
            'qty': int(order.quantity),
            'side': order.side.value,
            'type': order.order_type.value,
            'time_in_force': order.time_in_force.value,
            'client_order_id': order.id
        }
        
        if order.price:
            broker_order['limit_price'] = order.price
        
        if order.stop_price:
            broker_order['stop_price'] = order.stop_price
        
        return broker_order
    
    async def _submit_to_broker(self, broker_order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit order to broker API"""
        try:
            # Submit order via Alpaca API
            response = self.api_client.submit_order(**broker_order)
            return {
                'id': response.id,
                'status': response.status,
                'filled_qty': float(response.filled_qty or 0),
                'filled_avg_price': float(response.filled_avg_price or 0)
            }
        except Exception as e:
            logger.error(f"Broker submission error: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            with self.lock:
                order = self.pending_orders.get(order_id)
                if not order:
                    return False
            
            if order.broker_order_id:
                self.api_client.cancel_order(order.broker_order_id)
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
            
            await self._notify_order_callbacks(order)
            
            with self.lock:
                del self.pending_orders[order_id]
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    async def _notify_order_callbacks(self, order: Order):
        """Notify order update callbacks"""
        for callback in self.order_callbacks.get(order.id, []):
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")
    
    async def _notify_fill_callbacks(self, fill: Fill):
        """Notify fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        if not self.execution_times:
            return {}
        
        times = list(self.execution_times)
        return {
            'avg_execution_time_ms': sum(times) / len(times),
            'min_execution_time_ms': min(times),
            'max_execution_time_ms': max(times),
            'p95_execution_time_ms': sorted(times)[int(len(times) * 0.95)],
            'total_orders': len(times)
        }


class FillProcessor:
    """Process order fills and update positions"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.fill_callbacks = []
        self.redis_client = None
        self.kafka_producer = None
    
    def initialize(self):
        """Initialize fill processor"""
        try:
            self.redis_client = redis.from_url(config.redis_url, decode_responses=True)
            
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
        except Exception as e:
            logger.error(f"Fill processor initialization error: {e}")
    
    def add_callback(self, callback: Callable[[Fill], None]):
        """Add fill callback"""
        self.fill_callbacks.append(callback)
    
    async def process_fill(self, fill: Fill):
        """Process order fill"""
        try:
            # Update portfolio
            self.portfolio.add_fill(fill)
            
            # Cache fill in Redis
            await self._cache_fill(fill)
            
            # Publish to Kafka
            await self._publish_fill(fill)
            
            # Notify callbacks
            for callback in self.fill_callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    logger.error(f"Fill callback error: {e}")
            
            logger.info(f"Fill processed: {fill.symbol} {fill.quantity}@{fill.price}")
            
        except Exception as e:
            logger.error(f"Fill processing error: {e}")
    
    async def _cache_fill(self, fill: Fill):
        """Cache fill in Redis"""
        try:
            if not self.redis_client:
                return
            
            key = f"fill:{fill.id}"
            value = {
                'id': fill.id,
                'order_id': fill.order_id,
                'symbol': fill.symbol,
                'side': fill.side.value,
                'quantity': fill.quantity,
                'price': fill.price,
                'timestamp': fill.timestamp.isoformat(),
                'commission': fill.commission,
                'exchange': fill.exchange,
                'execution_id': fill.execution_id
            }
            
            self.redis_client.hset(key, mapping=value)
            self.redis_client.expire(key, 86400)  # 24 hours TTL
            
        except Exception as e:
            logger.error(f"Redis fill caching error: {e}")
    
    async def _publish_fill(self, fill: Fill):
        """Publish fill to Kafka"""
        try:
            if not self.kafka_producer:
                return
            
            message = {
                'type': 'fill',
                'fill_id': fill.id,
                'order_id': fill.order_id,
                'symbol': fill.symbol,
                'side': fill.side.value,
                'quantity': fill.quantity,
                'price': fill.price,
                'timestamp': fill.timestamp.isoformat(),
                'commission': fill.commission
            }
            
            self.kafka_producer.send('trading_events', value=message)
            
        except Exception as e:
            logger.error(f"Kafka fill publishing error: {e}")


class OrderManager:
    """Central order management system"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.validator = OrderValidator(portfolio)
        self.router = OrderRouter()
        self.execution_engine = None
        self.fill_processor = FillProcessor(portfolio)
        self.order_accepted_callbacks = []
        
        # Order tracking
        self.orders = {}  # order_id -> Order
        self.active_orders = {}  # order_id -> Order
        self.order_history = deque(maxlen=10000)
        
        # Performance tracking
        self.order_count = 0
        self.fill_count = 0
        self.start_time = time.time()
        
        # API client
        self.api_client = None
        
    async def initialize(self):
        """Initialize order manager"""
        try:
            # Initialize API client
            self.api_client = tradeapi.REST(
                key_id=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                base_url=config.ALPACA_BASE_URL
            )
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(self.api_client)
            self.execution_engine.add_fill_callback(self._on_fill)
            
            # Initialize fill processor
            self.fill_processor.initialize()
            
            # Add venues
            self.router.add_venue("alpaca", self.api_client)
            
            logger.info("Order manager initialized")
            
        except Exception as e:
            logger.error(f"Order manager initialization error: {e}")
            raise
    
    def add_order_accepted_callback(self, callback: Callable[[Order], None]):
        """Add a callback to be invoked when an order is accepted by the broker."""
        self.order_accepted_callbacks.append(callback)

    async def submit_order(self, order: Order) -> tuple[bool, str]:
        """Submit order for execution"""
        try:
            # Validate order
            is_valid, message = self.validator.validate_order(order)
            if not is_valid:
                logger.warning(f"Order validation failed: {message}")
                return False, message
            
            # Route order
            venue = self.router.route_order(order)
            logger.info(f"Order routed to venue: {venue}")
            
            # Store order
            self.orders[order.id] = order
            self.active_orders[order.id] = order
            self.order_history.append(order)
            self.order_count += 1
            
            # Submit for execution
            success = await self.execution_engine.submit_order(order)
            
            if success:
                logger.info(f"Order submitted successfully: {order.id}")
                
                # Notify that the order has been accepted by the broker
                order.on_submitted()
                for callback in self.order_accepted_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        logger.error(f"Order accepted callback error: {e}")

                return True, "Order submitted"
            else:
                # Remove from active orders if submission failed
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                return False, "Order submission failed"
                
        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return False, f"Submission error: {e}"
    
    async def cancel_order(self, order_id: str) -> tuple[bool, str]:
        """Cancel order"""
        try:
            if order_id not in self.active_orders:
                return False, "Order not found or not active"
            
            success = await self.execution_engine.cancel_order(order_id)
            
            if success:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                return True, "Order cancelled"
            else:
                return False, "Cancellation failed"
                
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False, f"Cancellation error: {e}"
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders (optionally for specific symbol)"""
        cancelled_count = 0
        orders_to_cancel = list(self.active_orders.keys())
        
        for order_id in orders_to_cancel:
            order = self.active_orders[order_id]
            if symbol is None or order.symbol == symbol:
                success, _ = await self.cancel_order(order_id)
                if success:
                    cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    def _on_fill(self, fill: Fill):
        """Handle fill notification"""
        try:
            self.fill_count += 1
            
            # Update order
            if fill.order_id in self.orders:
                order = self.orders[fill.order_id]
                order.update_fill(fill.quantity, fill.price)
                
                # Remove from active orders if filled
                if order.is_filled and order.id in self.active_orders:
                    del self.active_orders[order.id]
            
            # Process fill
            asyncio.create_task(self.fill_processor.process_fill(fill))
            
        except Exception as e:
            logger.error(f"Fill handling error: {e}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders (optionally filtered by symbol)"""
        if symbol:
            return [order for order in self.active_orders.values() 
                   if order.symbol == symbol]
        return list(self.active_orders.values())
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history"""
        orders = list(self.order_history)
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        return orders[-limit:]
    
    async def start_order_monitoring(self):
        """Start monitoring orders for updates"""
        while True:
            try:
                # Check for order updates from broker
                await self._check_order_updates()
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_order_updates(self):
        """
        Reconcile all active orders with the broker (Alpaca).
        This method acts as the source of truth for order state.
        """
        try:
            broker_orders = self.api_client.list_orders(status='open')
            broker_order_ids = {o.id for o in broker_orders}

            # 1. Update or add orders that are open at the broker
            for broker_order in broker_orders:
                order_id = broker_order.client_order_id
                
                if order_id in self.orders:
                    order = self.orders[order_id]
                    
                    # Update status if changed
                    new_status = OrderStatus(broker_order.status)
                    if order.status != new_status:
                        logger.info(f"Updating order {order_id} status: {order.status} -> {new_status}")
                        order.status = new_status

                    # Check for new fills
                    broker_filled_qty = float(broker_order.filled_qty or 0)
                    if broker_filled_qty > order.filled_quantity:
                        fill_qty = broker_filled_qty - order.filled_quantity
                        fill_price = float(broker_order.filled_avg_price or 0)
                        
                        fill = Fill(
                            order_id=order.id, symbol=order.symbol, side=order.side,
                            quantity=fill_qty, price=fill_price,
                            execution_id=broker_order.id
                        )
                        self._on_fill(fill)
                else:
                    # This is an order that exists on the broker but not in our memory.
                    # This is rare but could happen if the platform restarted.
                    # We will log it but won't add it back to a strategy automatically.
                    logger.warning(f"Found untracked open order at broker: {broker_order.id}",
                                   symbol=broker_order.symbol, qty=broker_order.qty)

            # 2. Remove local active orders that are no longer open at the broker
            local_active_ids = list(self.active_orders.keys())
            for order_id in local_active_ids:
                if self.orders[order_id].broker_order_id not in broker_order_ids:
                    logger.warning(f"Removing ghost active order {order_id} not found at broker.")
                    order = self.orders[order_id]
                    
                    # We need to find its final state from the broker
                    try:
                        final_broker_order = self.api_client.get_order(order.broker_order_id)
                        order.status = OrderStatus(final_broker_order.status)
                    except Exception:
                        # If it's not found at all, mark as cancelled as a safe default
                        order.status = OrderStatus.CANCELLED
                    
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
                        
        except Exception as e:
            logger.error(f"Order update check error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get order management statistics"""
        elapsed = time.time() - self.start_time
        
        stats = {
            'total_orders': self.order_count,
            'active_orders': len(self.active_orders),
            'total_fills': self.fill_count,
            'orders_per_second': self.order_count / elapsed if elapsed > 0 else 0,
            'uptime_seconds': elapsed
        }
        
        # Add execution stats
        if self.execution_engine:
            stats.update(self.execution_engine.get_execution_stats())
        
        return stats