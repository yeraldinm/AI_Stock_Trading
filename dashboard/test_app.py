"""
Simple test version of the trading dashboard to demonstrate strategy management
"""
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_dashboard_test_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Mock strategy data for testing
mock_strategies = {
    'MeanReversion_1': {
        'id': 'MeanReversion_1',
        'name': 'MeanReversion',
        'enabled': True,
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'parameters': {
            'lookback_period': 20,
            'deviation_threshold': 2.0,
            'max_position_size': 1000,
            'stop_loss': 0.02
        },
        'performance': {
            'total_pnl': 1250.75,
            'unrealized_pnl': 350.25,
            'realized_pnl': 900.50,
            'total_trades': 45,
            'winning_trades': 28,
            'win_rate': 0.622,
            'active_positions': 2,
            'active_orders': 1,
            'uptime_seconds': 3600,
            'signals_generated': 120
        },
        'positions': {
            'AAPL': {
                'symbol': 'AAPL',
                'quantity': 100,
                'average_price': 150.25,
                'market_value': 15250.0,
                'unrealized_pnl': 225.0,
                'realized_pnl': 0.0,
                'is_long': True,
                'is_short': False,
                'is_flat': False
            },
            'GOOGL': {
                'symbol': 'GOOGL',
                'quantity': 25,
                'average_price': 2800.50,
                'market_value': 70012.5,
                'unrealized_pnl': 125.25,
                'realized_pnl': 0.0,
                'is_long': True,
                'is_short': False,
                'is_flat': False
            }
        },
        'orders_count': 1,
        'last_update': datetime.now(timezone.utc).isoformat()
    },
    'MeanReversion_2': {
        'id': 'MeanReversion_2',
        'name': 'MeanReversion',
        'enabled': False,
        'symbols': ['TSLA', 'AMZN'],
        'parameters': {
            'lookback_period': 15,
            'deviation_threshold': 1.8,
            'max_position_size': 500,
            'stop_loss': 0.025
        },
        'performance': {
            'total_pnl': -125.50,
            'unrealized_pnl': 0.0,
            'realized_pnl': -125.50,
            'total_trades': 12,
            'winning_trades': 5,
            'win_rate': 0.417,
            'active_positions': 0,
            'active_orders': 0,
            'uptime_seconds': 1800,
            'signals_generated': 35
        },
        'positions': {},
        'orders_count': 0,
        'last_update': datetime.now(timezone.utc).isoformat()
    }
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/platform/status')
def get_platform_status():
    """Get mock platform status"""
    return jsonify({
        'running': True,
        'uptime_seconds': 7200,
        'stats': {
            'orders_placed': 57,
            'fills_processed': 52,
            'strategies_active': len([s for s in mock_strategies.values() if s['enabled']]),
            'total_pnl': sum(s['performance']['total_pnl'] for s in mock_strategies.values())
        }
    })

@app.route('/api/strategies')
def get_strategies():
    """Get list of all strategies and their status"""
    return jsonify(mock_strategies)

@app.route('/api/strategies/<strategy_id>/start', methods=['POST'])
def start_strategy(strategy_id):
    """Start a specific strategy"""
    if strategy_id in mock_strategies:
        mock_strategies[strategy_id]['enabled'] = True
        mock_strategies[strategy_id]['last_update'] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Strategy {strategy_id} started")
        
        # Emit update to all clients
        socketio.emit('strategy_status_update', mock_strategies)
        
        return jsonify({'status': 'success', 'message': f'Strategy {mock_strategies[strategy_id]["name"]} started'})
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/stop', methods=['POST'])
def stop_strategy(strategy_id):
    """Stop a specific strategy"""
    if strategy_id in mock_strategies:
        mock_strategies[strategy_id]['enabled'] = False
        mock_strategies[strategy_id]['last_update'] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Strategy {strategy_id} stopped")
        
        # Emit update to all clients
        socketio.emit('strategy_status_update', mock_strategies)
        
        return jsonify({'status': 'success', 'message': f'Strategy {mock_strategies[strategy_id]["name"]} stopped'})
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/restart', methods=['POST'])
def restart_strategy(strategy_id):
    """Restart a specific strategy"""
    if strategy_id in mock_strategies:
        mock_strategies[strategy_id]['enabled'] = False
        time.sleep(0.5)
        mock_strategies[strategy_id]['enabled'] = True
        mock_strategies[strategy_id]['last_update'] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Strategy {strategy_id} restarted")
        
        # Emit update to all clients
        socketio.emit('strategy_status_update', mock_strategies)
        
        return jsonify({'status': 'success', 'message': f'Strategy {mock_strategies[strategy_id]["name"]} restarted'})
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>/performance')
def get_strategy_performance_detail(strategy_id):
    """Get detailed performance data for a specific strategy"""
    if strategy_id in mock_strategies:
        strategy = mock_strategies[strategy_id]
        
        # Mock trade history
        trade_history = []
        for i in range(10):
            trade_history.append({
                'timestamp': (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
                'symbol': np.random.choice(strategy['symbols']),
                'side': np.random.choice(['buy', 'sell']),
                'quantity': np.random.randint(10, 100),
                'price': np.random.uniform(100, 300),
                'pnl': np.random.uniform(-50, 100)
            })
        
        performance_data = {
            'basic_metrics': strategy['performance'],
            'trade_history': trade_history,
            'pnl_history': [np.random.uniform(-100, 200) for _ in range(50)],
            'positions': strategy['positions'],
            'active_orders': [],
            'signals': []
        }
        return jsonify(performance_data)
    return jsonify({'error': 'Strategy not found'}), 404

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
        
        # Generate unique strategy ID
        strategy_id = f"{strategy_type}_{int(time.time())}"
        
        # Create new strategy
        mock_strategies[strategy_id] = {
            'id': strategy_id,
            'name': strategy_type,
            'enabled': True,
            'symbols': symbols,
            'parameters': parameters,
            'performance': {
                'total_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'active_positions': 0,
                'active_orders': 0,
                'uptime_seconds': 0,
                'signals_generated': 0
            },
            'positions': {},
            'orders_count': 0,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Created new strategy: {strategy_type} ({strategy_id})")
        
        # Emit update to all clients
        socketio.emit('strategy_status_update', mock_strategies)
        
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
    if strategy_id in mock_strategies:
        strategy_name = mock_strategies[strategy_id]['name']
        del mock_strategies[strategy_id]
        
        logger.info(f"Deleted strategy {strategy_id}")
        
        # Emit update to all clients
        socketio.emit('strategy_status_update', mock_strategies)
        
        return jsonify({'status': 'success', 'message': 'Strategy deleted successfully'})
    
    return jsonify({'status': 'error', 'message': 'Strategy not found'}), 404

@app.route('/api/market-data')
def get_market_data():
    """Get mock market data"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    market_data = {}
    
    for symbol in symbols:
        price = np.random.uniform(100, 300)
        change = np.random.uniform(-5, 5)
        market_data[symbol] = {
            'symbol': symbol,
            'price': price,
            'change': change,
            'change_percent': (change / price) * 100,
            'volume': np.random.randint(1000000, 10000000),
            'bid': price - 0.01,
            'ask': price + 0.01,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    return jsonify(market_data)

@app.route('/api/portfolio')
def get_portfolio():
    """Get mock portfolio data"""
    return jsonify({
        'total_value': 125000.0,
        'cash': 25000.0,
        'unrealized_pnl': 1250.75,
        'realized_pnl': 2750.25,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@app.route('/api/historical-data/<symbol>')
def get_historical_data(symbol):
    """Get historical price data for charting"""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=1)
    
    historical_data = []
    current_time = start_time
    current_price = np.random.uniform(100, 300)
    
    while current_time <= end_time:
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
        
        current_time += timedelta(minutes=5)
    
    return jsonify(historical_data)

@socketio.on('connect')
def on_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    
    # Send initial data to newly connected client
    emit('initial_data', {
        'market_data': {},
        'portfolio': {},
        'orders': [],
        'fills': [],
        'risk_metrics': {},
        'strategy_performance': {}
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

def update_mock_data():
    """Periodically update mock data and emit to clients"""
    import threading
    
    def update_loop():
        while True:
            time.sleep(5)
            
            # Update strategy performance randomly
            for strategy_id, strategy in mock_strategies.items():
                if strategy['enabled']:
                    # Simulate some trading activity
                    pnl_change = np.random.uniform(-10, 15)
                    strategy['performance']['total_pnl'] += pnl_change
                    strategy['performance']['unrealized_pnl'] += np.random.uniform(-5, 5)
                    
                    if np.random.random() > 0.8:  # 20% chance of new trade
                        strategy['performance']['total_trades'] += 1
                        if pnl_change > 0:
                            strategy['performance']['winning_trades'] += 1
                        
                        if strategy['performance']['total_trades'] > 0:
                            strategy['performance']['win_rate'] = (
                                strategy['performance']['winning_trades'] / 
                                strategy['performance']['total_trades']
                            )
                    
                    strategy['last_update'] = datetime.now(timezone.utc).isoformat()
            
            # Emit updates
            socketio.emit('strategy_status_update', mock_strategies)
    
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()

if __name__ == '__main__':
    # Start mock data updates
    update_mock_data()
    
    # Run the Flask-SocketIO app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)