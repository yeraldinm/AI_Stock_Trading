# Function Reference Guide

## Table of Contents
- [AlpacaPaperSocket Functions](#alpacapapersocket-functions)
- [TradingSystem Functions](#tradingsystem-functions)
- [PMModelDevelopment Functions](#pmmodeldevelopment-functions)
- [PortfolioManagementModel Functions](#portfoliomanagementmodel-functions)
- [PortfolioManagementSystem Functions](#portfoliomanagementsystem-functions)
- [Utility Functions](#utility-functions)

## AlpacaPaperSocket Functions

### `__init__()`

**Signature**: `def __init__(self)`

**Description**: Initializes the Alpaca Paper Trading API connection with hardcoded credentials.

**Parameters**: None

**Returns**: None

**Side Effects**: 
- Establishes connection to Alpaca paper trading API
- Sets up authentication using predefined credentials

**Example**:
```python
api = AlpacaPaperSocket()
```

**Inherited Methods** (from `alpaca_trade_api.REST`):

### `submit_order()`

**Signature**: `def submit_order(symbol, qty, side, type, time_in_force, **kwargs)`

**Description**: Submits a trading order to the Alpaca API.

**Parameters**:
- `symbol` (str): Stock symbol (e.g., 'IBM', 'AAPL')
- `qty` (int): Quantity of shares to trade
- `side` (str): Order side - 'buy' or 'sell'
- `type` (str): Order type - 'market', 'limit', 'stop', etc.
- `time_in_force` (str): Time in force - 'day', 'gtc', 'ioc', 'fok'
- `**kwargs`: Additional order parameters

**Returns**: Order object with order details

**Raises**:
- `APIError`: If order submission fails
- `ValueError`: If parameters are invalid

**Example**:
```python
order = api.submit_order(
    symbol='IBM',
    qty=10,
    side='buy',
    type='market',
    time_in_force='day'
)
```

### `get_barset()`

**Signature**: `def get_barset(symbols, timeframe, limit=100, start=None, end=None, **kwargs)`

**Description**: Retrieves historical price data for specified symbols.

**Parameters**:
- `symbols` (str or list): Stock symbol(s) to retrieve data for
- `timeframe` (str): Time frame ('1Min', '5Min', '15Min', '1H', '1D')
- `limit` (int): Maximum number of bars to return (default: 100)
- `start` (datetime, optional): Start date for data retrieval
- `end` (datetime, optional): End date for data retrieval

**Returns**: Barset object containing price data

**Example**:
```python
bars = api.get_barset('IBM', '1D', limit=30)
latest_close = bars['IBM'][-1].c
```

### `get_account()`

**Signature**: `def get_account()`

**Description**: Returns account information including buying power, equity, and positions.

**Parameters**: None

**Returns**: Account object with account details

**Example**:
```python
account = api.get_account()
buying_power = float(account.buying_power)
```

### `list_positions()`

**Signature**: `def list_positions()`

**Description**: Returns all open positions in the account.

**Parameters**: None

**Returns**: List of Position objects

**Example**:
```python
positions = api.list_positions()
for position in positions:
    print(f"{position.symbol}: {position.qty} shares")
```

## TradingSystem Functions

### `__init__()`

**Signature**: `def __init__(self, api, symbol, time_frame, system_id, system_label)`

**Description**: Initializes a trading system and starts the trading loop in a background thread.

**Parameters**:
- `api` (AlpacaPaperSocket): API connection object
- `symbol` (str): Stock symbol to trade
- `time_frame` (int): Time frame in seconds between trading decisions
- `system_id` (int): Unique identifier for this system instance
- `system_label` (str): Human-readable label for the system

**Returns**: None

**Side Effects**:
- Stores parameters as instance variables
- Creates and starts a background thread running `system_loop()`

**Example**:
```python
system = MyTradingSystem(
    api=AlpacaPaperSocket(),
    symbol='AAPL',
    time_frame=3600,
    system_id=1,
    system_label='Apple Trader'
)
```

### `place_buy_order()` (Abstract)

**Signature**: `@abc.abstractmethod def place_buy_order(self)`

**Description**: Abstract method that must be implemented by subclasses to execute buy orders.

**Parameters**: None

**Returns**: None (implementation dependent)

**Raises**: `NotImplementedError` if not overridden

**Implementation Example**:
```python
def place_buy_order(self):
    return self.api.submit_order(
        symbol=self.symbol,
        qty=1,
        side='buy',
        type='market',
        time_in_force='day'
    )
```

### `place_sell_order()` (Abstract)

**Signature**: `@abc.abstractmethod def place_sell_order(self)`

**Description**: Abstract method that must be implemented by subclasses to execute sell orders.

**Parameters**: None

**Returns**: None (implementation dependent)

**Raises**: `NotImplementedError` if not overridden

**Implementation Example**:
```python
def place_sell_order(self):
    return self.api.submit_order(
        symbol=self.symbol,
        qty=1,
        side='sell',
        type='market',
        time_in_force='day'
    )
```

### `system_loop()` (Abstract)

**Signature**: `@abc.abstractmethod def system_loop(self)`

**Description**: Abstract method containing the main trading logic loop. Runs in a separate thread.

**Parameters**: None

**Returns**: None

**Raises**: `NotImplementedError` if not overridden

**Implementation Example**:
```python
def system_loop(self):
    while True:
        # Trading logic here
        if should_buy():
            self.place_buy_order()
        elif should_sell():
            self.place_sell_order()
        time.sleep(self.time_frame)
```

## PMModelDevelopment Functions

### `__init__()`

**Signature**: `def __init__(self)`

**Description**: Develops, trains, and saves a neural network model for stock trading predictions.

**Parameters**: None

**Returns**: None

**Side Effects**:
- Reads data from 'IBM.csv'
- Creates and trains a neural network
- Saves model architecture to 'model.json'
- Saves model weights to 'weights.h5'
- Prints classification report to console

**Data Processing Steps**:
1. Load CSV data with columns 'Delta Close' and 'Signal'
2. Split into features (X) and labels (y)
3. Perform train-test split
4. Create 5-layer neural network
5. Train for 100 epochs
6. Evaluate and print results
7. Save model and weights

**Model Architecture**:
```
Input Layer:    1 neuron  (tanh activation)
Hidden Layer 1: 3 neurons (tanh activation)
Hidden Layer 2: 3 neurons (tanh activation)
Hidden Layer 3: 3 neurons (tanh activation)
Output Layer:   1 neuron  (tanh activation)
```

**Training Configuration**:
- Optimizer: RMSprop
- Loss: Hinge loss
- Metrics: Accuracy
- Epochs: 100

**File Dependencies**:
- Input: `IBM.csv` (must exist)
- Output: `model.json`, `weights.h5`

**Example**:
```python
# Train a new model
model_dev = PMModelDevelopment()
# Check console for training results
# Files model.json and weights.h5 will be created
```

**Expected CSV Format**:
```csv
Delta Close,Signal
-2.5,-1
1.8,1
0.3,0
...
```

## PortfolioManagementModel Functions

### `__init__()`

**Signature**: `def __init__(self)`

**Description**: Loads a pre-trained neural network model and validates it against test data.

**Parameters**: None

**Returns**: None

**Attributes Created**:
- `self.network`: Loaded Keras model ready for predictions

**Side Effects**:
- Loads model structure from 'model.json'
- Loads model weights from 'weights.h5'
- Validates model using test data from 'IBM.csv'
- Prints classification report to console

**File Dependencies**:
- `model.json`: Neural network architecture
- `weights.h5`: Trained model weights
- `IBM.csv`: Validation data

**Validation Process**:
1. Load test data from CSV
2. Make predictions on all test samples
3. Round predictions to nearest integer
4. Compare with actual labels
5. Print classification report

**Example**:
```python
# Load existing model
ai_model = PortfolioManagementModel()

# Model is now ready for predictions
prediction = ai_model.network.predict([[price_delta]])
```

**Error Handling**:
```python
try:
    ai_model = PortfolioManagementModel()
except FileNotFoundError as e:
    print(f"Required file missing: {e}")
except Exception as e:
    print(f"Model loading failed: {e}")
```

### `network.predict()` (Inherited from Keras)

**Signature**: `def predict(self, x, **kwargs)`

**Description**: Makes predictions using the loaded neural network.

**Parameters**:
- `x` (array-like): Input features, shape (n_samples, n_features)
- `**kwargs`: Additional Keras prediction parameters

**Returns**: numpy array of predictions, shape (n_samples, 1)

**Example**:
```python
# Single prediction
delta_close = 2.5
prediction = ai_model.network.predict([[delta_close]])

# Multiple predictions
deltas = [[1.2], [-0.8], [3.1]]
predictions = ai_model.network.predict(deltas)

# Interpret results
for i, pred in enumerate(predictions):
    if np.around(pred) <= -0.5:
        print(f"Sample {i}: SELL")
    elif np.around(pred) >= 0.5:
        print(f"Sample {i}: BUY")
    else:
        print(f"Sample {i}: HOLD")
```

## PortfolioManagementSystem Functions

### `__init__()`

**Signature**: `def __init__(self)`

**Description**: Initializes the complete AI trading system with predefined configuration.

**Parameters**: None

**Returns**: None

**Configuration**:
- API: AlpacaPaperSocket()
- Symbol: 'IBM'
- Time frame: 86400 seconds (24 hours)
- System ID: 1
- System label: 'AI_PM'

**Attributes Created**:
- `self.AI`: Instance of PortfolioManagementModel
- All parent class attributes from TradingSystem

**Side Effects**:
- Starts background trading thread
- Loads AI model for predictions

**Example**:
```python
# Start automated trading
trading_system = PortfolioManagementSystem()
# System runs automatically in background
```

### `place_buy_order()`

**Signature**: `def place_buy_order(self)`

**Description**: Places a market buy order for 1 share of IBM.

**Parameters**: None

**Returns**: None

**Order Details**:
- Symbol: 'IBM'
- Quantity: 1 share
- Side: 'buy'
- Type: 'market'
- Time in force: 'day'

**Example**:
```python
system = PortfolioManagementSystem()
system.place_buy_order()  # Manually trigger buy order
```

**Error Handling**:
```python
def place_buy_order(self):
    try:
        self.api.submit_order(
            symbol='IBM',
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print("Buy order placed successfully")
    except Exception as e:
        print(f"Buy order failed: {e}")
```

### `place_sell_order()`

**Signature**: `def place_sell_order(self)`

**Description**: Places a market sell order for 1 share of IBM.

**Parameters**: None

**Returns**: None

**Order Details**:
- Symbol: 'IBM'
- Quantity: 1 share
- Side: 'sell'
- Type: 'market'
- Time in force: 'day'

**Example**:
```python
system = PortfolioManagementSystem()
system.place_sell_order()  # Manually trigger sell order
```

### `system_loop()`

**Signature**: `def system_loop(self)`

**Description**: Main trading loop that runs continuously in a background thread.

**Parameters**: None

**Returns**: None (runs indefinitely)

**Loop Logic**:
1. Wait 24 hours (1440 minutes)
2. Request daily price data for IBM
3. Track weekly price changes
4. Every 7 days, calculate price delta
5. Use AI model to predict trading action
6. Execute buy/sell orders based on prediction

**Trading Decision Logic**:
```python
if np.around(prediction) <= -0.5:
    self.place_sell_order()  # Bearish signal
elif np.around(prediction) >= 0.5:
    self.place_buy_order()   # Bullish signal
# Otherwise: Hold (no action)
```

**State Variables**:
- `this_weeks_close`: Current week's closing price
- `last_weeks_close`: Previous week's closing price
- `delta`: Weekly price change
- `day_count`: Days elapsed in current week

**Data Flow**:
```
Daily Close Price → Weekly Delta → AI Prediction → Trading Action
```

**Example Execution Flow**:
```
Day 1-6: Collect daily prices, no trading
Day 7:   Calculate weekly delta
         Feed to AI model
         Execute trade if signal is strong enough
         Reset counter for next week
```

## Utility Functions

### Data Validation Functions

```python
def validate_csv_data(filename):
    """
    Validates that CSV file has required columns and data format.
    
    Parameters:
        filename (str): Path to CSV file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        data = pd.read_csv(filename)
        required_columns = ['Delta Close', 'Signal']
        return all(col in data.columns for col in required_columns)
    except:
        return False

def validate_prediction(prediction):
    """
    Validates AI model prediction output.
    
    Parameters:
        prediction (numpy.ndarray): Model prediction
        
    Returns:
        bool: True if valid prediction, False otherwise
    """
    if prediction is None:
        return False
    if np.isnan(prediction).any():
        return False
    return True
```

### Configuration Helper Functions

```python
def load_config(config_file='config.json'):
    """
    Loads system configuration from JSON file.
    
    Parameters:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
    """
    import json
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return get_default_config()

def get_default_config():
    """
    Returns default system configuration.
    
    Returns:
        dict: Default configuration parameters
    """
    return {
        'symbol': 'IBM',
        'time_frame': 86400,
        'quantity': 1,
        'model_file': 'model.json',
        'weights_file': 'weights.h5'
    }
```

### Logging Helper Functions

```python
def setup_logging(log_level='INFO'):
    """
    Sets up logging configuration for the trading system.
    
    Parameters:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_system.log'),
            logging.StreamHandler()
        ]
    )

def log_trade_action(action, symbol, quantity, price=None):
    """
    Logs trading actions for audit trail.
    
    Parameters:
        action (str): Trading action ('BUY', 'SELL', 'HOLD')
        symbol (str): Stock symbol
        quantity (int): Number of shares
        price (float, optional): Execution price
    """
    import logging
    logger = logging.getLogger(__name__)
    message = f"Action: {action}, Symbol: {symbol}, Quantity: {quantity}"
    if price:
        message += f", Price: ${price:.2f}"
    logger.info(message)
```

### Performance Monitoring Functions

```python
def calculate_portfolio_performance(start_date, end_date):
    """
    Calculates portfolio performance metrics.
    
    Parameters:
        start_date (datetime): Start date for calculation
        end_date (datetime): End date for calculation
        
    Returns:
        dict: Performance metrics (return, volatility, sharpe_ratio)
    """
    # Implementation would require trade history data
    pass

def get_system_health_status():
    """
    Checks system health and returns status report.
    
    Returns:
        dict: System health metrics
    """
    import os
    health_status = {
        'model_files_exist': all(os.path.exists(f) for f in ['model.json', 'weights.h5']),
        'data_file_exists': os.path.exists('IBM.csv'),
        'api_connection': test_api_connection(),
        'timestamp': time.time()
    }
    return health_status

def test_api_connection():
    """
    Tests API connection and returns status.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        api = AlpacaPaperSocket()
        api.get_account()
        return True
    except:
        return False
```

---

*This function reference provides detailed documentation for all public methods and utility functions in the AI Stock Trading System. Each function includes signature, parameters, return values, examples, and error handling guidance.*