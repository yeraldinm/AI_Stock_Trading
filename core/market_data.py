"""
High-performance market data handler for low latency trading
"""
import asyncio
import json
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any, Set
import logging
import inspect

import aiohttp
import websockets
import redis
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer

from .models import (
    Tick, OHLCV, OrderBook, MarketData, 
    OrderSide, OrderStatus, OrderType
)
from config import config

logger = logging.getLogger(__name__)


class MarketDataBuffer:
    """High-performance circular buffer for market data"""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.RLock()
        self._index = {}  # symbol -> list of indices
    
    def add(self, data: MarketData):
        """Add market data to buffer"""
        with self.lock:
            self.buffer.append(data)
            
            # Update index
            if data.symbol not in self._index:
                self._index[data.symbol] = deque(maxlen=1000)
            self._index[data.symbol].append(len(self.buffer) - 1)
    
    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        with self.lock:
            if symbol not in self._index or not self._index[symbol]:
                return None
            
            latest_idx = self._index[symbol][-1]
            if latest_idx < len(self.buffer):
                return self.buffer[latest_idx]
        return None
    
    def get_history(self, symbol: str, count: int = 100) -> List[MarketData]:
        """Get historical market data for symbol"""
        with self.lock:
            if symbol not in self._index:
                return []
            
            indices = list(self._index[symbol])[-count:]
            return [self.buffer[idx] for idx in indices if idx < len(self.buffer)]


class WebSocketDataFeed:
    """WebSocket market data feed handler"""
    
    def __init__(self, url: str, symbols: List[str]):
        self.url = url
        self.symbols = symbols
        self.websocket = None
        self.running = False
        self.callbacks = []
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10
        self.reconnect_count = 0
        
    def add_callback(self, callback: Callable[[MarketData], None]):
        """Add callback for market data updates"""
        self.callbacks.append(callback)
    
    async def connect(self):
        """Connect to WebSocket feed"""
        try:
            logger.debug(f"Attempting to connect to WebSocket: {self.url} with symbols: {self.symbols}")
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            logger.info(f"Connected to WebSocket feed: {self.url}")
            self.reconnect_count = 0
            # Alpaca: Wait for 'connected', then authenticate, then subscribe
            if "alpaca" in self.url:
                await self._alpaca_handshake_and_auth()
            else:
                await self._subscribe()
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}", exc_info=True)
            logger.debug(f"WebSocket connection parameters: url={self.url}, symbols={self.symbols}")
            raise
    
    async def _alpaca_handshake_and_auth(self):
        """Wait for 'connected', then authenticate, then subscribe for Alpaca WebSocket."""
        # Wait for the initial 'connected' message
        try:
            logger.debug("Waiting for Alpaca 'connected' message...")
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            logger.debug(f"Received Alpaca handshake response: {response}")
            data = json.loads(response)
            if isinstance(data, list):
                found = False
                for msg in data:
                    if msg.get("T") == "success" and msg.get("msg", "") == "connected":
                        found = True
                        break
                if not found:
                    logger.error(f"Did not receive expected 'connected' message from Alpaca: {data}")
                    raise Exception(f"Did not receive expected 'connected' message from Alpaca: {data}")
            elif isinstance(data, dict):
                if not (data.get("T") == "success" and data.get("msg", "") == "connected"):
                    logger.error(f"Did not receive expected 'connected' message from Alpaca: {data}")
                    raise Exception(f"Did not receive expected 'connected' message from Alpaca: {data}")
            logger.info("Alpaca WebSocket handshake successful. Proceeding to authenticate.")
        except Exception as e:
            logger.error(f"Exception during Alpaca WebSocket handshake: {e}", exc_info=True)
            raise
        # Authenticate
        await self._authenticate_alpaca()
        # Subscribe
        await self._subscribe()
    
    async def _authenticate_alpaca(self):
        """Authenticate with Alpaca WebSocket API before subscribing."""
        from config import config
        auth_msg = {
            "action": "auth",
            "key": config.ALPACA_API_KEY,
            "secret": config.ALPACA_SECRET_KEY
        }
        logger.debug(f"Sending Alpaca auth message: {auth_msg}")
        await self.websocket.send(json.dumps(auth_msg))
        # Wait for auth response
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            logger.debug(f"Received Alpaca auth response: {response}")
            data = json.loads(response)
            if isinstance(data, list):
                for msg in data:
                    if msg.get("T") == "error":
                        logger.error(f"Alpaca WebSocket auth error: {msg}")
                        raise Exception(f"Alpaca WebSocket auth error: {msg}")
                    if msg.get("T") == "success" and msg.get("msg", "") == "authenticated":
                        logger.info("Alpaca WebSocket authenticated successfully.")
                        return
            elif isinstance(data, dict):
                if data.get("T") == "error":
                    logger.error(f"Alpaca WebSocket auth error: {data}")
                    raise Exception(f"Alpaca WebSocket auth error: {data}")
                if data.get("T") == "success" and data.get("msg", "") == "authenticated":
                    logger.info("Alpaca WebSocket authenticated successfully.")
                    return
            logger.error(f"Unexpected Alpaca auth response: {data}")
            raise Exception(f"Unexpected Alpaca auth response: {data}")
        except Exception as e:
            logger.error(f"Exception during Alpaca WebSocket authentication: {e}", exc_info=True)
            raise
    
    async def _subscribe(self):
        """Subscribe to market data for symbols"""
        subscribe_msg = {
            "action": "subscribe",
            "trades": self.symbols,
            "quotes": self.symbols,
            "bars": self.symbols
        }
        await self.websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to symbols: {self.symbols}")
    
    async def start(self):
        """Start the WebSocket data feed"""
        self.running = True
        
        while self.running:
            try:
                if not self.websocket:
                    await self.connect()
                
                async for message in self.websocket:
                    if not self.running:
                        break
                    
                    try:
                        logger.debug(f"Received WebSocket message: {message}")
                        data = json.loads(message)
                        await self._process_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message: {e}")
                        logger.debug(f"Raw message: {message}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}", exc_info=True)
                        logger.debug(f"Raw message: {message}")
                        
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                logger.debug(f"ConnectionClosed details: code={getattr(e, 'code', None)}, reason={getattr(e, 'reason', None)}")
                await self._handle_reconnect()
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await self._handle_reconnect()
    
    async def _handle_reconnect(self):
        """Handle WebSocket reconnection"""
        if not self.running:
            return
            
        logger.debug(f"Handling reconnect: reconnect_count={self.reconnect_count}, max_reconnect_attempts={self.max_reconnect_attempts}")
        self.websocket = None
        self.reconnect_count += 1
        
        if self.reconnect_count > self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.running = False
            return
        
        logger.info(f"Reconnecting in {self.reconnect_delay} seconds (attempt {self.reconnect_count})")
        await asyncio.sleep(self.reconnect_delay)
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        if not isinstance(data, list):
            return
        
        for item in data:
            message_type = item.get('T')
            symbol = item.get('S')
            
            if not symbol:
                continue
            
            market_data = None
            
            if message_type == 'q':  # Quote
                market_data = self._parse_quote(item)
            elif message_type == 't':  # Trade
                market_data = self._parse_trade(item)
            elif message_type == 'b':  # Bar
                market_data = self._parse_bar(item)
            
            if market_data:
                # Call all callbacks
                for callback in self.callbacks:
                    try:
                        if inspect.iscoroutinefunction(callback):
                            asyncio.create_task(callback(market_data))
                        else:
                            callback(market_data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
    
    def _parse_quote(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse quote message"""
        try:
            t_val = data['t']
            timestamp = None
            if isinstance(t_val, int):
                timestamp = datetime.fromtimestamp(t_val / 1000, timezone.utc)
            else:
                try:
                    t_val_int = int(t_val)
                    timestamp = datetime.fromtimestamp(t_val_int / 1000, timezone.utc)
                except Exception:
                    # Try ISO8601 string
                    try:
                        iso_str = t_val.rstrip('Z')
                        timestamp = datetime.fromisoformat(iso_str)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        logger.debug(f"Parsed ISO8601 quote timestamp: {t_val} -> {timestamp}")
                    except Exception as e:
                        logger.error(f"Failed to parse quote timestamp: {t_val}, error: {e}")
                        return None
            tick = Tick(
                symbol=data['S'],
                timestamp=timestamp,
                bid=float(data['bp']),
                ask=float(data['ap']),
                bid_size=int(data['bs']),
                ask_size=int(data['as'])
            )
            
            return MarketData(
                symbol=tick.symbol,
                timestamp=tick.timestamp,
                tick=tick
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse quote: {e}")
            return None
    
    def _parse_trade(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse trade message"""
        try:
            t_val = data['t']
            timestamp = None
            if isinstance(t_val, int):
                timestamp = datetime.fromtimestamp(t_val / 1000, timezone.utc)
            else:
                try:
                    t_val_int = int(t_val)
                    timestamp = datetime.fromtimestamp(t_val_int / 1000, timezone.utc)
                except Exception:
                    # Try ISO8601 string
                    try:
                        iso_str = t_val.rstrip('Z')
                        timestamp = datetime.fromisoformat(iso_str)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        logger.debug(f"Parsed ISO8601 trade timestamp: {t_val} -> {timestamp}")
                    except Exception as e:
                        logger.error(f"Failed to parse trade timestamp: {t_val}, error: {e}")
                        return None
            tick = Tick(
                symbol=data['S'],
                timestamp=timestamp,
                bid=0.0,  # Will be updated by quote
                ask=0.0,  # Will be updated by quote
                bid_size=0,
                ask_size=0,
                last_price=float(data['p']),
                last_size=int(data['s']),
                volume=int(data.get('v', 0))
            )
            
            return MarketData(
                symbol=tick.symbol,
                timestamp=tick.timestamp,
                tick=tick
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse trade: {e}")
            return None
    
    def _parse_bar(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse bar message"""
        try:
            t_val = data['t']
            timestamp = None
            if isinstance(t_val, int):
                timestamp = datetime.fromtimestamp(t_val / 1000, timezone.utc)
            else:
                try:
                    t_val_int = int(t_val)
                    timestamp = datetime.fromtimestamp(t_val_int / 1000, timezone.utc)
                except Exception:
                    # Try ISO8601 string
                    try:
                        iso_str = t_val.rstrip('Z')
                        timestamp = datetime.fromisoformat(iso_str)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        logger.debug(f"Parsed ISO8601 bar timestamp: {t_val} -> {timestamp}")
                    except Exception as e:
                        logger.error(f"Failed to parse bar timestamp: {t_val}, error: {e}")
                        return None
            ohlcv = OHLCV(
                symbol=data['S'],
                timestamp=timestamp,
                open=float(data['o']),
                high=float(data['h']),
                low=float(data['l']),
                close=float(data['c']),
                volume=int(data['v']),
                timeframe="1m"
            )
            return MarketData(
                symbol=ohlcv.symbol,
                timestamp=ohlcv.timestamp,
                ohlcv=ohlcv
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse bar: {e}")
            return None
    
    async def stop(self):
        """Stop the WebSocket feed"""
        self.running = False
        if self.websocket:
            await self.websocket.close()


class MarketDataManager:
    """Central market data manager"""
    
    def __init__(self):
        self.buffer = MarketDataBuffer(maxsize=config.TICK_BUFFER_SIZE)
        self.feeds = {}
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.callbacks = defaultdict(list)
        self.running = False
        self.price_cache = {}  # symbol -> latest price
        self.last_update = {}  # symbol -> timestamp
        
        # Performance metrics
        self.message_count = 0
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize market data manager"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(config.redis_url, decode_responses=True)
            await self._test_redis_connection()
            
            # Initialize Kafka
            self._initialize_kafka()
            
            logger.info("Market data manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize market data manager: {e}")
            raise
    
    async def _test_redis_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def _initialize_kafka(self):
        """Initialize Kafka producer and consumer"""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                batch_size=16384,
                linger_ms=1,
                compression_type='snappy'
            )
            
            self.kafka_consumer = KafkaConsumer(
                'market_data',
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                group_id=config.KAFKA_CONSUMER_GROUP,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                enable_auto_commit=True,
                auto_offset_reset='latest'
            )
            
            logger.info("Kafka initialized")
            
        except Exception as e:
            logger.error(f"Kafka initialization failed: {e}")
    
    def add_feed(self, name: str, feed: WebSocketDataFeed):
        """Add market data feed"""
        self.feeds[name] = feed
        feed.add_callback(self._on_market_data)
    
    def subscribe(self, symbol: str, callback: Callable[[MarketData], None]):
        """Subscribe to market data updates for symbol"""
        self.callbacks[symbol].append(callback)
        logger.info(f"Subscribed to {symbol}")
    
    def unsubscribe(self, symbol: str, callback: Callable[[MarketData], None]):
        """Unsubscribe from market data updates"""
        if symbol in self.callbacks and callback in self.callbacks[symbol]:
            self.callbacks[symbol].remove(callback)
    
    def _on_market_data(self, data: MarketData):
        """Handle incoming market data"""
        try:
            # Update message count
            self.message_count += 1
            
            # Add to buffer
            self.buffer.add(data)
            
            # Update price cache
            if data.price:
                self.price_cache[data.symbol] = data.price
                self.last_update[data.symbol] = data.timestamp
            
            # Cache in Redis
            self._cache_in_redis(data)
            
            # Publish to Kafka
            self._publish_to_kafka(data)
            
            # Call symbol-specific callbacks
            for callback in self.callbacks.get(data.symbol, []):
                try:
                    if inspect.iscoroutinefunction(callback):
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Callback error for {data.symbol}: {e}")
            
            # Call global callbacks
            for callback in self.callbacks.get('*', []):
                try:
                    if inspect.iscoroutinefunction(callback):
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Global callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _cache_in_redis(self, data: MarketData):
        """Cache market data in Redis"""
        try:
            if not self.redis_client:
                return
            # Cache latest tick data
            if data.tick:
                key = f"tick:{data.symbol}"
                value = {
                    'symbol': data.tick.symbol or '',
                    'timestamp': data.tick.timestamp.isoformat() if data.tick.timestamp else '',
                    'bid': data.tick.bid if data.tick.bid is not None else 0.0,
                    'ask': data.tick.ask if data.tick.ask is not None else 0.0,
                    'bid_size': data.tick.bid_size if data.tick.bid_size is not None else 0,
                    'ask_size': data.tick.ask_size if data.tick.ask_size is not None else 0,
                    'last_price': data.tick.last_price if data.tick.last_price is not None else 0.0,
                    'last_size': data.tick.last_size if data.tick.last_size is not None else 0,
                    'volume': data.tick.volume if data.tick.volume is not None else 0
                }
                for k, v in value.items():
                    if v is None:
                        logger.warning(f"Redis tick cache: field {k} is None for symbol {data.symbol}")
                        value[k] = '' if isinstance(v, str) else 0
                self.redis_client.hset(key, mapping=value)
                self.redis_client.expire(key, 3600)  # 1 hour TTL
            # Cache OHLCV data
            if data.ohlcv:
                key = f"ohlcv:{data.symbol}:{data.ohlcv.timeframe}"
                value = {
                    'symbol': data.ohlcv.symbol or '',
                    'timestamp': data.ohlcv.timestamp.isoformat() if data.ohlcv.timestamp else '',
                    'open': data.ohlcv.open if data.ohlcv.open is not None else 0.0,
                    'high': data.ohlcv.high if data.ohlcv.high is not None else 0.0,
                    'low': data.ohlcv.low if data.ohlcv.low is not None else 0.0,
                    'close': data.ohlcv.close if data.ohlcv.close is not None else 0.0,
                    'volume': data.ohlcv.volume if data.ohlcv.volume is not None else 0,
                    'timeframe': data.ohlcv.timeframe or ''
                }
                for k, v in value.items():
                    if v is None:
                        logger.warning(f"Redis OHLCV cache: field {k} is None for symbol {data.symbol}")
                        value[k] = '' if isinstance(v, str) else 0
                self.redis_client.hset(key, mapping=value)
                self.redis_client.expire(key, 86400)  # 24 hours TTL
        except Exception as e:
            logger.error(f"Redis caching error: {e}")
    
    def _publish_to_kafka(self, data: MarketData):
        """Publish market data to Kafka"""
        try:
            if not self.kafka_producer:
                return
            
            message = {
                'symbol': data.symbol,
                'timestamp': data.timestamp.isoformat(),
                'type': 'tick' if data.tick else 'ohlcv' if data.ohlcv else 'orderbook'
            }
            
            if data.tick:
                message['tick'] = {
                    'bid': data.tick.bid,
                    'ask': data.tick.ask,
                    'bid_size': data.tick.bid_size,
                    'ask_size': data.tick.ask_size,
                    'last_price': data.tick.last_price,
                    'last_size': data.tick.last_size,
                    'volume': data.tick.volume
                }
            
            if data.ohlcv:
                message['ohlcv'] = {
                    'open': data.ohlcv.open,
                    'high': data.ohlcv.high,
                    'low': data.ohlcv.low,
                    'close': data.ohlcv.close,
                    'volume': data.ohlcv.volume,
                    'timeframe': data.ohlcv.timeframe
                }
            
            self.kafka_producer.send('market_data', value=message)
            
        except Exception as e:
            logger.error(f"Kafka publishing error: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.price_cache.get(symbol)
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        return self.buffer.get_latest(symbol)
    
    def get_history(self, symbol: str, count: int = 100) -> List[MarketData]:
        """Get historical market data for symbol"""
        return self.buffer.get_history(symbol, count)
    
    async def start(self):
        """Start all market data feeds"""
        self.running = True
        
        # Start all feeds
        tasks = []
        for name, feed in self.feeds.items():
            task = asyncio.create_task(feed.start())
            tasks.append(task)
            logger.info(f"Started feed: {name}")
        
        # Start Kafka consumer
        if self.kafka_consumer:
            consumer_task = asyncio.create_task(self._consume_kafka())
            tasks.append(consumer_task)
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in market data feeds: {e}")
    
    async def _consume_kafka(self):
        """Consume messages from Kafka"""
        try:
            for message in self.kafka_consumer:
                if not self.running:
                    break
                
                # Process Kafka message if needed
                # This can be used for distributed market data processing
                pass
                
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
    
    async def stop(self):
        """Stop all market data feeds"""
        self.running = False
        
        # Stop all feeds
        for name, feed in self.feeds.items():
            await feed.stop()
            logger.info(f"Stopped feed: {name}")
        
        # Close connections
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.kafka_consumer:
            self.kafka_consumer.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        elapsed = time.time() - self.start_time
        return {
            'messages_processed': self.message_count,
            'messages_per_second': self.message_count / elapsed if elapsed > 0 else 0,
            'uptime_seconds': elapsed,
            'active_symbols': len(self.price_cache),
            'buffer_size': len(self.buffer.buffer),
            'feeds_count': len(self.feeds)
        }