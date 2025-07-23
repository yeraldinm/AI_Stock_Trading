# AI Stock Trading System - API Documentation

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## Overview

The AI Stock Trading System is a Python-based automated trading platform that uses machine learning to make buy/sell decisions for stock trading. The system integrates with Alpaca's paper trading API and uses neural networks for decision-making.

### Key Features
- Abstract trading system framework
- AI-powered portfolio management
- Neural network-based decision making
- Alpaca API integration for paper trading
- Automated weekly trading cycles

## Core Components

### 1. AlpacaPaperSocket
A connection wrapper for Alpaca's paper trading API.

### 2. TradingSystem (Abstract Base Class)
The foundation class for all trading systems, providing a common interface and threading model.

### 3. PMModelDevelopment
Handles the development and training of the AI model using historical data.

### 4. PortfolioManagementModel
Loads and uses the trained AI model for making trading predictions.

### 5. PortfolioManagementSystem
The complete trading system that combines AI predictions with actual trading operations.

## API Reference

### AlpacaPaperSocket

```python
class AlpacaPaperSocket(REST)
```

**Description**: Extends Alpaca's REST API client with pre-configured paper trading credentials.

**Inherits from**: `alpaca_trade_api.REST`

**Constructor**:
```python
def __init__(self)
```

**Usage**:
```python
api = AlpacaPaperSocket()
# Now you can use all Alpaca REST API methods
```

**Methods**: Inherits all methods from Alpaca's REST API including:
- `submit_order()` - Submit trading orders
- `get_barset()` - Get historical price data
- `get_account()` - Get account information
- `list_positions()` - Get current positions

---

### TradingSystem (Abstract Base Class)

```python
class TradingSystem(abc.ABC)
```

**Description**: Abstract base class that defines the interface for all trading systems. Automatically starts a background thread for the trading loop.

**Constructor**:
```python
def __init__(self, api, symbol, time_frame, system_id, system_label)
```

**Parameters**:
- `api` (AlpacaPaperSocket): The API connection object
- `symbol` (str): Stock symbol to trade (e.g., 'IBM')
- `time_frame` (int): Time frame in seconds (e.g., 86400 for daily)
- `system_id` (int): Unique identifier for the system
- `system_label` (str): Human-readable label for the system

**Abstract Methods** (must be implemented by subclasses):

#### place_buy_order()
```python
@abc.abstractmethod
def place_buy_order(self)
```
**Description**: Execute a buy order for the configured symbol.
**Returns**: None
**Raises**: Must be implemented by subclass

#### place_sell_order()
```python
@abc.abstractmethod
def place_sell_order(self)
```
**Description**: Execute a sell order for the configured symbol.
**Returns**: None
**Raises**: Must be implemented by subclass

#### system_loop()
```python
@abc.abstractmethod
def system_loop(self)
```
**Description**: Main trading logic loop that runs in a separate thread.
**Returns**: None
**Raises**: Must be implemented by subclass

**Usage Example**:
```python
class MyTradingSystem(TradingSystem):
    def __init__(self):
        super().__init__(
            api=AlpacaPaperSocket(),
            symbol='AAPL',
            time_frame=86400,
            system_id=1,
            system_label='My System'
        )
    
    def place_buy_order(self):
        self.api.submit_order(
            symbol=self.symbol,
            qty=10,
            side='buy',
            type='market',
            time_in_force='day'
        )
    
    def place_sell_order(self):
        self.api.submit_order(
            symbol=self.symbol,
            qty=10,
            side='sell',
            type='market',
            time_in_force='day'
        )
    
    def system_loop(self):
        while True:
            # Your trading logic here
            time.sleep(self.time_frame)
```

---

### PMModelDevelopment

```python
class PMModelDevelopment
```

**Description**: Handles the development, training, and saving of the AI model using historical stock data.

**Constructor**:
```python
def __init__(self)
```

**Behavior**:
- Reads historical data from 'IBM.csv'
- Creates and trains a neural network model
- Saves the model structure to 'model.json'
- Saves the model weights to 'weights.h5'
- Prints classification report for model evaluation

**Data Requirements**:
- CSV file named 'IBM.csv' with columns:
  - `Delta Close`: Feature representing price change
  - `Signal`: Target variable for buy/sell/hold decisions

**Model Architecture**:
- Input layer: 1 neuron (tanh activation)
- Hidden layers: 3 layers with 3 neurons each (tanh activation)
- Output layer: 1 neuron (tanh activation)
- Optimizer: RMSprop
- Loss function: Hinge loss
- Training epochs: 100

**Usage**:
```python
# Train and save a new model
model_dev = PMModelDevelopment()
# This will create model.json and weights.h5 files
```

**Output Files**:
- `model.json`: Neural network architecture in JSON format
- `weights.h5`: Trained model weights in HDF5 format

---

### PortfolioManagementModel

```python
class PortfolioManagementModel
```

**Description**: Loads a pre-trained AI model and provides prediction capabilities for trading decisions.

**Constructor**:
```python
def __init__(self)
```

**Attributes**:
- `network`: Loaded Keras neural network model

**Behavior**:
- Loads model structure from 'model.json'
- Loads model weights from 'weights.h5'
- Validates the model by making predictions on test data
- Prints classification report for validation

**Usage**:
```python
# Load existing trained model
ai_model = PortfolioManagementModel()

# Make predictions
delta_close = 5.2  # Price change
prediction = ai_model.network.predict([[delta_close]])

# Interpret prediction
if np.around(prediction) <= -0.5:
    decision = "SELL"
elif np.around(prediction) >= 0.5:
    decision = "BUY"
else:
    decision = "HOLD"
```

**Required Files**:
- `model.json`: Model architecture file
- `weights.h5`: Model weights file
- `IBM.csv`: Test data for validation

---

### PortfolioManagementSystem

```python
class PortfolioManagementSystem(TradingSystem)
```

**Description**: Complete AI-powered trading system that combines machine learning predictions with automated trading execution.

**Constructor**:
```python
def __init__(self)
```

**Inherits from**: `TradingSystem`

**Attributes**:
- `AI`: Instance of `PortfolioManagementModel`
- All attributes from parent `TradingSystem` class

**Configuration**:
- Symbol: 'IBM'
- Time frame: 86400 seconds (daily)
- System ID: 1
- System label: 'AI_PM'

**Methods**:

#### place_buy_order()
```python
def place_buy_order(self)
```
**Description**: Places a market buy order for 1 share of IBM.
**Returns**: None

#### place_sell_order()
```python
def place_sell_order(self)
```
**Description**: Places a market sell order for 1 share of IBM.
**Returns**: None

#### system_loop()
```python
def system_loop(self)
```
**Description**: Main trading loop that:
1. Waits for daily market data
2. Calculates weekly price changes
3. Uses AI model to make trading decisions
4. Executes buy/sell orders based on predictions

**Trading Logic**:
- Collects daily closing prices
- Every 7 days, calculates weekly delta (price change)
- Feeds delta to AI model for prediction
- If prediction ≤ -0.5: Execute sell order
- If prediction ≥ 0.5: Execute buy order
- Otherwise: Hold (no action)

**Usage**:
```python
# Start the automated trading system
trading_system = PortfolioManagementSystem()
# System will run automatically in background thread
```

## Usage Examples

### Complete Workflow Example

```python
# Step 1: Develop and train the AI model
print("Training AI model...")
model_dev = PMModelDevelopment()
print("Model training complete. Files saved: model.json, weights.h5")

# Step 2: Test the trained model
print("Loading and testing model...")
ai_model = PortfolioManagementModel()
print("Model loaded and validated successfully")

# Step 3: Start automated trading
print("Starting automated trading system...")
trading_system = PortfolioManagementSystem()
print("Trading system started. Check logs for trading activity.")

# The system will now run automatically in the background
```

### Custom Trading System Example

```python
class CustomTradingSystem(TradingSystem):
    def __init__(self, symbol, quantity):
        super().__init__(
            api=AlpacaPaperSocket(),
            symbol=symbol,
            time_frame=3600,  # 1 hour
            system_id=2,
            system_label=f'Custom_{symbol}'
        )
        self.quantity = quantity
        self.ai_model = PortfolioManagementModel()
    
    def place_buy_order(self):
        try:
            self.api.submit_order(
                symbol=self.symbol,
                qty=self.quantity,
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"Buy order placed: {self.quantity} shares of {self.symbol}")
        except Exception as e:
            print(f"Buy order failed: {e}")
    
    def place_sell_order(self):
        try:
            self.api.submit_order(
                symbol=self.symbol,
                qty=self.quantity,
                side='sell',
                type='market',
                time_in_force='day'
            )
            print(f"Sell order placed: {self.quantity} shares of {self.symbol}")
        except Exception as e:
            print(f"Sell order failed: {e}")
    
    def system_loop(self):
        while True:
            try:
                # Get latest price data
                bars = self.api.get_barset(self.symbol, '1H', limit=2)
                if len(bars[self.symbol]) >= 2:
                    current_price = bars[self.symbol][-1].c
                    previous_price = bars[self.symbol][-2].c
                    delta = current_price - previous_price
                    
                    # Get AI prediction
                    prediction = self.ai_model.network.predict([[delta]])
                    
                    if np.around(prediction) <= -0.5:
                        self.place_sell_order()
                    elif np.around(prediction) >= 0.5:
                        self.place_buy_order()
                
                time.sleep(self.time_frame)
            except Exception as e:
                print(f"System loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

# Usage
custom_system = CustomTradingSystem('AAPL', 5)
```

## Configuration

### Required Files
- `IBM.csv`: Historical stock data with columns 'Delta Close' and 'Signal'
- `model.json`: (Generated) Neural network architecture
- `weights.h5`: (Generated) Trained model weights

### Environment Variables
The system uses hardcoded Alpaca paper trading credentials. For production use, consider using environment variables:

```python
import os

class AlpacaPaperSocket(REST):
    def __init__(self):
        super().__init__(
            key_id=os.getenv('ALPACA_KEY_ID', 'PKPO0ZH3XTVB336B7TEO'),
            secret_key=os.getenv('ALPACA_SECRET_KEY', 'gcs4U2Hp/ACI4A5UwYjYugrPqB2odD/m40Zuz5qw'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )
```

### Dependencies
```bash
pip install pandas numpy keras scikit-learn alpaca-trade-api
```

## Error Handling

### Common Issues and Solutions

1. **Missing Data Files**
   ```python
   # Check if required files exist
   import os
   required_files = ['IBM.csv', 'model.json', 'weights.h5']
   for file in required_files:
       if not os.path.exists(file):
           print(f"Error: {file} not found")
   ```

2. **API Connection Issues**
   ```python
   try:
       api = AlpacaPaperSocket()
       account = api.get_account()
   except Exception as e:
       print(f"API connection failed: {e}")
   ```

3. **Model Loading Issues**
   ```python
   try:
       ai_model = PortfolioManagementModel()
   except FileNotFoundError:
       print("Model files not found. Run PMModelDevelopment() first.")
   except Exception as e:
       print(f"Model loading failed: {e}")
   ```

### Best Practices

1. **Always validate data before trading**:
   ```python
   def validate_prediction(prediction):
       if prediction is None or np.isnan(prediction):
           return False
       return True
   ```

2. **Implement proper logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   def place_buy_order(self):
       try:
           # Place order
           logger.info(f"Buy order placed for {self.symbol}")
       except Exception as e:
           logger.error(f"Buy order failed: {e}")
   ```

3. **Use position sizing and risk management**:
   ```python
   def calculate_position_size(self, account_value, risk_percentage=0.02):
       return int(account_value * risk_percentage / current_price)
   ```

## Testing

The system includes test utilities in `tests/test_trading_system.py`. To run tests:

```bash
cd tests
python -m pytest test_trading_system.py -v
```

The test framework provides:
- Module loading without executing demo code
- Mock implementations for external dependencies
- Thread behavior validation

## Support and Maintenance

For optimal performance:
1. Regularly retrain the AI model with fresh data
2. Monitor system performance and adjust parameters
3. Keep dependencies updated
4. Implement proper error handling and logging
5. Test thoroughly in paper trading before live deployment

---

*This documentation covers version 1.0 of the AI Stock Trading System. For updates and additional features, please refer to the project repository.*