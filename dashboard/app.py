"""
Advanced Trading Dashboard - Main Application
"""
import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import redis
import numpy as np
from dataclasses import asdict
import sys
import os
import atexit

# Add parent directory to path to import trading platform modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_platform import TradingPlatform
from config import config
from core.models import MarketData, Order, Fill, Portfolio
from core.market_data import MarketDataManager
from core.order_management import OrderManager
from core.risk_management import RiskManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_dashboard_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Trading platform instance
platform: Optional[TradingPlatform] = None
platform_thread: Optional[threading.Thread] = None
platform_lock = threading.Lock()


# Redis connection for real-time data
redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
    decode_responses=True
)

class DashboardDataManager:
    """Manages real-time data for the dashboard"""
    
    def __init__(self):
        self.market_data_cache = {}
        self.portfolio_cache = {}
        self.orders_cache = []
        self.fills_cache = []
        self.risk_metrics_cache = {}
        self.strategy_performance_cache = {}
        self.connected_clients = set()
        
    def update_market_data(self, symbol: str, data: Dict[str, Any]):
        """Update market data cache and broadcast to clients"""
        self.market_data_cache[symbol] = {
            **data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol
        }
        
        # Broadcast to all connected clients
        socketio.emit('market_data_update', {
            'symbol': symbol,
            'data': self.market_data_cache[symbol]
        })
        
    def update_portfolio(self, portfolio_data: Dict[str, Any]):
        """Update portfolio cache and broadcast"""
        self.portfolio_cache = {
            **portfolio_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        socketio.emit('portfolio_update', self.portfolio_cache)
        
    def add_order(self, order_data: Dict[str, Any]):
        """Add new order and broadcast"""
        order_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self.orders_cache.append(order_data)
        
        # Keep only last 1000 orders
        if len(self.orders_cache) > 1000:
            self.orders_cache = self.orders_cache[-1000:]
            
        socketio.emit('new_order', order_data)
        
    def add_fill(self, fill_data: Dict[str, Any]):
        """Add new fill and broadcast"""
        fill_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self.fills_cache.append(fill_data)
        
        # Keep only last 1000 fills
        if len(self.fills_cache) > 1000:
            self.fills_cache = self.fills_cache[-1000:]
            
        socketio.emit('new_fill', fill_data)
        
    def update_risk_metrics(self, risk_data: Dict[str, Any]):
        """Update risk metrics and broadcast"""
        self.risk_metrics_cache = {
            **risk_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        socketio.emit('risk_metrics_update', self.risk_metrics_cache)
        
    def update_strategy_performance(self, strategy_name: str, performance_data: Dict[str, Any]):
        """Update strategy performance and broadcast"""
        self.strategy_performance_cache[strategy_name] = {
            **performance_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        socketio.emit('strategy_performance_update', {
            'strategy': strategy_name,
            'data': self.strategy_performance_cache[strategy_name]
        })

# Initialize dashboard data manager
dashboard_manager = DashboardDataManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/platform/start', methods=['POST'])
def start_platform():
    """Start the trading platform"""
    global platform, platform_thread
    with platform_lock:
        if platform is None:
            platform = TradingPlatform()
            
            def run_platform():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(platform.initialize())
                loop.run_until_complete(platform.start())

            platform_thread = threading.Thread(target=run_platform, daemon=True)
            platform_thread.start()
            
            logger.info("Trading platform started in background thread")
            return jsonify({'status': 'success', 'message': 'Platform starting'})
        
        return jsonify({'status': 'warning', 'message': 'Platform is already running or starting'})

@app.route('/api/platform/stop', methods=['POST'])
def stop_platform():
    """Stop the trading platform"""
    global platform, platform_thread
    with platform_lock:
        if platform and platform.running:
            
            async def do_stop():
                await platform.stop()

            # Run the async stop function in the platform's event loop
            stop_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(stop_loop)
            stop_loop.run_until_complete(do_stop())

            platform_thread.join(timeout=10)
            
            platform = None
            platform_thread = None
            
            logger.info("Trading platform stopped")
            return jsonify({'status': 'success', 'message': 'Platform stopped'})
        
        return jsonify({'status': 'warning', 'message': 'Platform is not running'})

@app.route('/api/platform/status')
def get_platform_status():
    """Get the status of the trading platform"""
    if platform:
        return jsonify(platform.get_platform_status())
    
    return jsonify({'running': False})
    
@app.route('/api/market-data')
def get_market_data():
    """Get current market data for all symbols"""
    return jsonify(dashboard_manager.market_data_cache)

@app.route('/api/portfolio')
def get_portfolio():
    """Get current portfolio data"""
    return jsonify(dashboard_manager.portfolio_cache)

@app.route('/api/orders')
def get_orders():
    """Get recent orders"""
    return jsonify(dashboard_manager.orders_cache[-100:])  # Last 100 orders

@app.route('/api/fills')
def get_fills():
    """Get recent fills"""
    return jsonify(dashboard_manager.fills_cache[-100:])  # Last 100 fills

@app.route('/api/risk-metrics')
def get_risk_metrics():
    """Get current risk metrics"""
    return jsonify(dashboard_manager.risk_metrics_cache)

@app.route('/api/strategy-performance')
def get_strategy_performance():
    """Get strategy performance data"""
    return jsonify(dashboard_manager.strategy_performance_cache)

@app.route('/api/strategies')
def get_strategies():
    """Get list of all strategies and their status"""
    if platform and platform.strategies:
        strategies_data = {}
        for strategy_id, strategy in platform.strategies.items():
            strategies_data[strategy_id] = {
                'id': strategy_id,
                'name': strategy.name,
                'enabled': strategy.enabled,
                'symbols': strategy.symbols,
                'parameters': strategy.parameters,
                'performance': strategy.get_performance_summary(),
                'positions': {symbol: pos.to_dict() for symbol, pos in strategy.positions.items()},
                'orders_count': len(strategy.orders),
                'last_update': strategy.last_update.isoformat() if strategy.last_update else None
            }
        return jsonify(strategies_data)
    return jsonify({})

@app.route('/api/strategies/<strategy_id>/start', methods=['POST'])
def start_strategy(strategy_id):
    """Start a specific strategy"""
    if platform and strategy_id in platform.strategies:
        strategy = platform.strategies[strategy_id]
        strategy.enabled = True
        logger.info(f"Strategy {strategy_id} started")
        return jsonify({'status': 'success', 'message': f'Strategy {strategy.name} started'})
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/stop', methods=['POST'])
def stop_strategy(strategy_id):
    """Stop a specific strategy"""
    if platform and strategy_id in platform.strategies:
        strategy = platform.strategies[strategy_id]
        strategy.enabled = False
        logger.info(f"Strategy {strategy_id} stopped")
        return jsonify({'status': 'success', 'message': f'Strategy {strategy.name} stopped'})
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/restart', methods=['POST'])
def restart_strategy(strategy_id):
    """Restart a specific strategy"""
    if platform and strategy_id in platform.strategies:
        strategy = platform.strategies[strategy_id]
        strategy.enabled = False
        # Wait a moment then re-enable
        import time
        time.sleep(0.5)
        strategy.enabled = True
        logger.info(f"Strategy {strategy_id} restarted")
        return jsonify({'status': 'success', 'message': f'Strategy {strategy.name} restarted'})
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/performance')
def get_strategy_performance_detail(strategy_id):
    """Get detailed performance data for a specific strategy"""
    if platform and strategy_id in platform.strategies:
        strategy = platform.strategies[strategy_id]
        performance_data = {
            'basic_metrics': strategy.get_performance_summary(),
            'trade_history': list(strategy.trade_history)[-100:],  # Last 100 trades
            'pnl_history': list(strategy.pnl_history),
            'positions': {symbol: pos.to_dict() for symbol, pos in strategy.positions.items()},
            'active_orders': [order.to_dict() for order in strategy.orders.values() if order.status in ['PENDING', 'PARTIALLY_FILLED']],
            'signals': list(strategy.signals)[-50:] if hasattr(strategy, 'signals') else []
        }
        return jsonify(performance_data)
    return jsonify({'error': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/positions')
def get_strategy_positions(strategy_id):
    """Get positions for a specific strategy"""
    if platform and strategy_id in platform.strategies:
        strategy = platform.strategies[strategy_id]
        positions_data = {
            symbol: {
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'average_price': pos.average_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'is_flat': pos.is_flat
            }
            for symbol, pos in strategy.positions.items()
            if not pos.is_flat
        }
        return jsonify(positions_data)
    return jsonify({})

@app.route('/api/strategies/<strategy_id>/trades')
def get_strategy_trades(strategy_id):
    """Get trade history for a specific strategy"""
    if platform and strategy_id in platform.strategies:
        strategy = platform.strategies[strategy_id]
        trades_data = []
        
        # Get recent trades from trade history
        for trade in list(strategy.trade_history)[-100:]:  # Last 100 trades
            if hasattr(trade, 'to_dict'):
                trades_data.append(trade.to_dict())
            else:
                trades_data.append(trade)
        
        return jsonify(trades_data)
    return jsonify([])

@app.route('/api/strategies/create', methods=['POST'])
def create_strategy():
    """Create a new strategy instance"""
    try:
        data = request.json
        strategy_type = data.get('type')
        symbols = data.get('symbols', [])
        parameters = data.get('parameters', {})
        
        if not strategy_type or not symbols:
            return jsonify({'error': 'Strategy type and symbols are required'}), 400
        
        if not platform:
            return jsonify({'error': 'Trading platform is not running'}), 400
        
        # Generate unique strategy ID
        import time
        strategy_id = f"{strategy_type}_{int(time.time())}"
        
        # Create strategy based on type
        if strategy_type == "MeanReversion":
            from strategies.mean_reversion import MeanReversionStrategy
            strategy = MeanReversionStrategy(
                strategy_id=strategy_id,
                symbols=symbols,
                parameters=parameters
            )
        else:
            return jsonify({'error': f'Unknown strategy type: {strategy_type}'}), 400
        
        # Set up strategy callbacks
        strategy.add_order_callback(platform._on_strategy_order)
        strategy.add_signal_callback(platform._on_strategy_signal)
        
        # Add to platform
        platform.strategies[strategy_id] = strategy
        platform.stats['strategies_active'] = len(platform.strategies)
        
        logger.info(f"Created new strategy: {strategy_type} ({strategy_id})")
        
        return jsonify({
            'status': 'success',
            'strategy_id': strategy_id,
            'message': f'Strategy {strategy_type} created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<strategy_id>/delete', methods=['DELETE'])
def delete_strategy(strategy_id):
    """Delete a strategy"""
    if platform and strategy_id in platform.strategies:
        strategy = platform.strategies[strategy_id]
        
        # Stop the strategy first
        strategy.enabled = False
        
        # Cancel any active orders
        # Note: In a real implementation, you'd want to cancel orders through the order manager
        
        # Remove from platform
        del platform.strategies[strategy_id]
        platform.stats['strategies_active'] = len(platform.strategies)
        
        logger.info(f"Deleted strategy {strategy_id}")
        return jsonify({'status': 'success', 'message': 'Strategy deleted successfully'})
    
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/historical-data/<symbol>')
def get_historical_data(symbol):
    """Get historical price data for charting"""
    # Generate sample historical data (in production, this would come from your data source)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)
    
    # Sample data generation (replace with actual data retrieval)
    import datetime
    current_time = start_time
    base_price = 150.0  # Sample base price
    
    historical_data = []
    current_price = base_price
    
    while current_time <= end_time:
        # Simple random walk for demonstration
        change = np.random.normal(0, 0.5)
        current_price += change
        
        historical_data.append({
            'timestamp': current_time.isoformat(),
            'open': current_price,
            'high': current_price + abs(np.random.normal(0, 0.3)),
            'low': current_price - abs(np.random.normal(0, 0.3)),
            'close': current_price,
            'volume': np.random.randint(1000, 10000)
        })
        
        current_time += timedelta(hours=1)
    
    return jsonify(historical_data)

@app.route('/api/place-order', methods=['POST'])
def place_order():
    """Place a new order"""
    try:
        order_data = request.json
        
        # Validate required fields
        required_fields = ['symbol', 'side', 'quantity', 'order_type']
        for field in required_fields:
            if field not in order_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Add order to cache (in production, this would go through the order management system)
        order = {
            'id': f"ORDER_{int(time.time() * 1000)}",
            'symbol': order_data['symbol'],
            'side': order_data['side'],
            'quantity': float(order_data['quantity']),
            'order_type': order_data['order_type'],
            'price': float(order_data.get('price', 0)),
            'status': 'PENDING',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        dashboard_manager.add_order(order)
        
        return jsonify({'success': True, 'order': order})
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    dashboard_manager.connected_clients.add(request.sid)
    logger.info(f"Client connected: {request.sid}")
    
    # Send initial data to newly connected client
    emit('initial_data', {
        'market_data': dashboard_manager.market_data_cache,
        'portfolio': dashboard_manager.portfolio_cache,
        'orders': dashboard_manager.orders_cache[-50:],  # Last 50 orders
        'fills': dashboard_manager.fills_cache[-50:],    # Last 50 fills
        'risk_metrics': dashboard_manager.risk_metrics_cache,
        'strategy_performance': dashboard_manager.strategy_performance_cache
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    dashboard_manager.connected_clients.discard(request.sid)
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('subscribe_symbol')
def on_subscribe_symbol(data):
    """Subscribe to updates for a specific symbol"""
    symbol = data.get('symbol')
    if symbol:
        join_room(f'symbol_{symbol}')
        logger.info(f"Client {request.sid} subscribed to {symbol}")

@socketio.on('unsubscribe_symbol')
def on_unsubscribe_symbol(data):
    """Unsubscribe from updates for a specific symbol"""
    symbol = data.get('symbol')
    if symbol:
        leave_room(f'symbol_{symbol}')
        logger.info(f"Client {request.sid} unsubscribed from {symbol}")

def listen_to_redis():
    """Listen to Redis pub/sub for real-time updates from the trading platform"""
    pubsub = redis_client.pubsub()
    pubsub.subscribe({
        'market_data': lambda m: dashboard_manager.update_market_data(m['data']),
        'portfolio_updates': lambda m: dashboard_manager.update_portfolio(json.loads(m['data'])),
        'order_updates': lambda m: dashboard_manager.add_order(json.loads(m['data'])),
        'fill_updates': lambda m: dashboard_manager.add_fill(json.loads(m['data'])),
        'risk_updates': lambda m: dashboard_manager.update_risk_metrics(json.loads(m['data'])),
        'strategy_updates': lambda m: dashboard_manager.update_strategy_performance(
            json.loads(m['data'])['strategy'], 
            json.loads(m['data'])['data']
        )
    })
    
    logger.info("Subscribed to Redis channels for real-time updates")
    pubsub.run_in_thread(sleep_time=0.01)

def platform_status_emitter():
    """Periodically emit platform status to clients"""
    while True:
        try:
            status = {}
            if platform:
                status = platform.get_platform_status()
                
                # Also emit strategy updates
                if platform.strategies:
                    strategies_data = {}
                    for strategy_id, strategy in platform.strategies.items():
                        strategies_data[strategy_id] = {
                            'id': strategy_id,
                            'name': strategy.name,
                            'enabled': strategy.enabled,
                            'symbols': strategy.symbols,
                            'parameters': strategy.parameters,
                            'performance': strategy.get_performance_summary(),
                            'positions': {symbol: pos.to_dict() for symbol, pos in strategy.positions.items()},
                            'orders_count': len(strategy.orders),
                            'last_update': strategy.last_update.isoformat() if strategy.last_update else None
                        }
                    socketio.emit('strategy_status_update', strategies_data)
            else:
                status = {'running': False}
                
            socketio.emit('platform_status_update', status)
            
        except Exception as e:
            logger.error(f"Error emitting platform status: {e}")
            
        time.sleep(5)

def cleanup_platform():
    """Ensure graceful shutdown of the platform when the app exits"""
    if platform and platform.running:
        logger.info("Flask app is shutting down, stopping platform...")
        
        async def do_stop():
            await platform.stop()
            
        stop_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(stop_loop)
        stop_loop.run_until_complete(do_stop())

if __name__ == '__main__':
    # Start Redis listener
    listen_to_redis()
    
    # Start platform status emitter
    status_thread = threading.Thread(target=platform_status_emitter, daemon=True)
    status_thread.start()
    
    # Register shutdown hook
    atexit.register(cleanup_platform)
    
    # Run the Flask-SocketIO app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)