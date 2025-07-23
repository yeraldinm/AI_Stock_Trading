# AI Stock Trading System - Usage Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Step-by-Step Tutorial](#step-by-step-tutorial)
- [Advanced Usage](#advanced-usage)
- [Customization Examples](#customization-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## Quick Start

### Prerequisites
```bash
# Install required dependencies
pip install pandas numpy keras scikit-learn alpaca-trade-api tensorflow
```

### 5-Minute Setup
```python
# 1. Import the system
from AI_Stock_Trading import PMModelDevelopment, PortfolioManagementSystem

# 2. Train the AI model (one-time setup)
print("Training AI model...")
model_dev = PMModelDevelopment()

# 3. Start automated trading
print("Starting trading system...")
trading_system = PortfolioManagementSystem()
print("System is now running!")
```

That's it! Your AI trading system is now running automatically in the background.

## Step-by-Step Tutorial

### Step 1: Prepare Your Data

First, ensure you have the required data file `IBM.csv` with the correct format:

```csv
Delta Close,Signal
-2.5,-1
1.8,1
0.3,0
-1.2,-1
3.4,1
```

**Data Requirements:**
- `Delta Close`: Price change values (float)
- `Signal`: Trading signals (-1 for sell, 0 for hold, 1 for buy)

**Creating Sample Data:**
```python
import pandas as pd
import numpy as np

# Generate sample training data
np.random.seed(42)
n_samples = 1000

# Generate price deltas
deltas = np.random.normal(0, 2, n_samples)

# Generate signals based on deltas (simple strategy)
signals = np.where(deltas > 1, 1, np.where(deltas < -1, -1, 0))

# Create DataFrame
data = pd.DataFrame({
    'Delta Close': deltas,
    'Signal': signals
})

# Save to CSV
data.to_csv('IBM.csv', index=False)
print("Sample data created: IBM.csv")
```

### Step 2: Develop and Train Your AI Model

```python
from AI_Stock_Trading import PMModelDevelopment

# Initialize model development
print("Starting model development...")
model_dev = PMModelDevelopment()

# The system will:
# 1. Load data from IBM.csv
# 2. Split into training and testing sets
# 3. Create a neural network
# 4. Train for 100 epochs
# 5. Evaluate performance
# 6. Save model.json and weights.h5

print("Model development complete!")
print("Files created: model.json, weights.h5")
```

**Expected Output:**
```
Epoch 1/100
...
Epoch 100/100

              precision    recall  f1-score   support

        -1.0       0.85      0.87      0.86       85
         0.0       0.92      0.91      0.91      150
         1.0       0.88      0.86      0.87       65

    accuracy                           0.89      300
   macro avg       0.88      0.88      0.88      300
weighted avg       0.89      0.89      0.89      300

Model development complete!
```

### Step 3: Test the Trained Model

```python
from AI_Stock_Trading import PortfolioManagementModel
import numpy as np

# Load the trained model
print("Loading trained model...")
ai_model = PortfolioManagementModel()

# Test with sample predictions
test_deltas = [2.5, -1.8, 0.3, -3.2, 1.1]

print("\nTesting model predictions:")
for i, delta in enumerate(test_deltas):
    prediction = ai_model.network.predict([[delta]])
    rounded_pred = np.around(prediction[0][0])
    
    if rounded_pred <= -0.5:
        action = "SELL"
    elif rounded_pred >= 0.5:
        action = "BUY"
    else:
        action = "HOLD"
    
    print(f"Delta: {delta:6.1f} → Prediction: {prediction[0][0]:6.3f} → Action: {action}")
```

**Expected Output:**
```
Loading trained model...

Testing model predictions:
Delta:    2.5 → Prediction:  0.847 → Action: BUY
Delta:   -1.8 → Prediction: -0.723 → Action: SELL
Delta:    0.3 → Prediction:  0.124 → Action: HOLD
Delta:   -3.2 → Prediction: -0.891 → Action: SELL
Delta:    1.1 → Prediction:  0.456 → Action: HOLD
```

### Step 4: Start Automated Trading

```python
from AI_Stock_Trading import PortfolioManagementSystem

# Start the complete trading system
print("Initializing AI trading system...")
trading_system = PortfolioManagementSystem()

print("Trading system started!")
print("The system will:")
print("- Run continuously in the background")
print("- Collect daily IBM price data")
print("- Make trading decisions weekly")
print("- Execute buy/sell orders automatically")

# The system is now running in a background thread
# You can continue with other tasks or let it run
```

### Step 5: Monitor System Activity

```python
import time
import logging

# Set up logging to monitor system activity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

# Check system health
def check_system_health():
    import os
    health_checks = {
        'Model files exist': os.path.exists('model.json') and os.path.exists('weights.h5'),
        'Data file exists': os.path.exists('IBM.csv'),
        'Log file exists': os.path.exists('trading_system.log')
    }
    
    print("\nSystem Health Check:")
    for check, status in health_checks.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{check}: {status_str}")

# Run health check
check_system_health()

# Monitor for a few minutes
print("\nMonitoring system (press Ctrl+C to stop)...")
try:
    while True:
        time.sleep(60)  # Check every minute
        print(f"System running... {time.strftime('%Y-%m-%d %H:%M:%S')}")
except KeyboardInterrupt:
    print("\nMonitoring stopped.")
```

## Advanced Usage

### Custom Trading Strategies

#### 1. Multi-Symbol Trading System

```python
from AI_Stock_Trading import TradingSystem, AlpacaPaperSocket, PortfolioManagementModel
import time
import numpy as np

class MultiSymbolTradingSystem(TradingSystem):
    def __init__(self, symbols, quantities):
        # Initialize with first symbol (required by parent)
        super().__init__(
            api=AlpacaPaperSocket(),
            symbol=symbols[0],
            time_frame=3600,  # 1 hour
            system_id=2,
            system_label='Multi_Symbol_AI'
        )
        
        self.symbols = symbols
        self.quantities = dict(zip(symbols, quantities))
        self.ai_model = PortfolioManagementModel()
        
    def place_buy_order(self, symbol=None):
        symbol = symbol or self.symbol
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=self.quantities[symbol],
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"✓ Buy order placed: {self.quantities[symbol]} shares of {symbol}")
            return order
        except Exception as e:
            print(f"✗ Buy order failed for {symbol}: {e}")
            return None
    
    def place_sell_order(self, symbol=None):
        symbol = symbol or self.symbol
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=self.quantities[symbol],
                side='sell',
                type='market',
                time_in_force='day'
            )
            print(f"✓ Sell order placed: {self.quantities[symbol]} shares of {symbol}")
            return order
        except Exception as e:
            print(f"✗ Sell order failed for {symbol}: {e}")
            return None
    
    def get_price_delta(self, symbol):
        """Calculate price delta for a symbol"""
        try:
            bars = self.api.get_barset(symbol, '1H', limit=2)
            if len(bars[symbol]) >= 2:
                current = bars[symbol][-1].c
                previous = bars[symbol][-2].c
                return current - previous
        except Exception as e:
            print(f"Error getting price data for {symbol}: {e}")
        return 0
    
    def system_loop(self):
        print(f"Starting multi-symbol trading for: {', '.join(self.symbols)}")
        
        while True:
            try:
                for symbol in self.symbols:
                    print(f"\nAnalyzing {symbol}...")
                    
                    # Get price delta
                    delta = self.get_price_delta(symbol)
                    if delta == 0:
                        continue
                    
                    # Get AI prediction
                    prediction = self.ai_model.network.predict([[delta]])
                    decision_value = np.around(prediction[0][0])
                    
                    print(f"{symbol} - Delta: {delta:.2f}, Prediction: {prediction[0][0]:.3f}")
                    
                    # Make trading decision
                    if decision_value <= -0.5:
                        self.place_sell_order(symbol)
                    elif decision_value >= 0.5:
                        self.place_buy_order(symbol)
                    else:
                        print(f"→ HOLD {symbol}")
                
                print(f"\nWaiting {self.time_frame} seconds until next analysis...")
                time.sleep(self.time_frame)
                
            except Exception as e:
                print(f"System loop error: {e}")
                time.sleep(60)

# Usage
symbols = ['IBM', 'AAPL', 'GOOGL', 'MSFT']
quantities = [10, 5, 2, 8]
multi_system = MultiSymbolTradingSystem(symbols, quantities)
```

#### 2. Risk-Managed Trading System

```python
class RiskManagedTradingSystem(TradingSystem):
    def __init__(self, symbol, max_position_size, stop_loss_pct):
        super().__init__(
            api=AlpacaPaperSocket(),
            symbol=symbol,
            time_frame=1800,  # 30 minutes
            system_id=3,
            system_label=f'Risk_Managed_{symbol}'
        )
        
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.ai_model = PortfolioManagementModel()
        self.entry_price = None
        self.current_position = 0
        
    def get_current_position(self):
        """Get current position for the symbol"""
        try:
            positions = self.api.list_positions()
            for position in positions:
                if position.symbol == self.symbol:
                    return int(position.qty)
        except Exception as e:
            print(f"Error getting position: {e}")
        return 0
    
    def calculate_position_size(self, account_value, risk_per_trade=0.02):
        """Calculate position size based on account value and risk"""
        try:
            current_price = self.get_current_price()
            if current_price:
                risk_amount = account_value * risk_per_trade
                position_size = int(risk_amount / (current_price * self.stop_loss_pct))
                return min(position_size, self.max_position_size)
        except Exception as e:
            print(f"Error calculating position size: {e}")
        return 1
    
    def get_current_price(self):
        """Get current price for the symbol"""
        try:
            bars = self.api.get_barset(self.symbol, '1Min', limit=1)
            return bars[self.symbol][-1].c
        except:
            return None
    
    def check_stop_loss(self):
        """Check if stop loss should be triggered"""
        if self.entry_price is None or self.current_position == 0:
            return False
        
        current_price = self.get_current_price()
        if current_price is None:
            return False
        
        if self.current_position > 0:  # Long position
            loss_pct = (self.entry_price - current_price) / self.entry_price
            if loss_pct >= self.stop_loss_pct:
                print(f"Stop loss triggered! Loss: {loss_pct:.2%}")
                return True
        else:  # Short position
            loss_pct = (current_price - self.entry_price) / self.entry_price
            if loss_pct >= self.stop_loss_pct:
                print(f"Stop loss triggered! Loss: {loss_pct:.2%}")
                return True
        
        return False
    
    def place_buy_order(self):
        account = self.api.get_account()
        account_value = float(account.equity)
        position_size = self.calculate_position_size(account_value)
        
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=position_size,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            self.entry_price = self.get_current_price()
            self.current_position = position_size
            print(f"✓ Risk-managed buy: {position_size} shares at ${self.entry_price:.2f}")
            return order
            
        except Exception as e:
            print(f"✗ Buy order failed: {e}")
    
    def place_sell_order(self):
        position_size = abs(self.current_position) or 1
        
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=position_size,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            current_price = self.get_current_price()
            if self.entry_price and current_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                print(f"✓ Sell order: {position_size} shares at ${current_price:.2f} (P&L: {pnl_pct:.2%})")
            
            self.entry_price = None
            self.current_position = 0
            return order
            
        except Exception as e:
            print(f"✗ Sell order failed: {e}")
    
    def system_loop(self):
        print(f"Starting risk-managed trading for {self.symbol}")
        print(f"Max position: {self.max_position_size}, Stop loss: {self.stop_loss_pct:.1%}")
        
        while True:
            try:
                # Update current position
                self.current_position = self.get_current_position()
                
                # Check stop loss first
                if self.check_stop_loss():
                    if self.current_position != 0:
                        self.place_sell_order()
                    continue
                
                # Get price data and make prediction
                bars = self.api.get_barset(self.symbol, '30Min', limit=2)
                if len(bars[self.symbol]) >= 2:
                    current_price = bars[self.symbol][-1].c
                    previous_price = bars[self.symbol][-2].c
                    delta = current_price - previous_price
                    
                    prediction = self.ai_model.network.predict([[delta]])
                    decision_value = np.around(prediction[0][0])
                    
                    print(f"Position: {self.current_position}, Delta: {delta:.2f}, Prediction: {prediction[0][0]:.3f}")
                    
                    # Trading logic with position management
                    if decision_value >= 0.5 and self.current_position <= 0:
                        if self.current_position < 0:
                            self.place_sell_order()  # Close short first
                        self.place_buy_order()
                    elif decision_value <= -0.5 and self.current_position >= 0:
                        if self.current_position > 0:
                            self.place_sell_order()  # Close long first
                
                time.sleep(self.time_frame)
                
            except Exception as e:
                print(f"System loop error: {e}")
                time.sleep(60)

# Usage
risk_system = RiskManagedTradingSystem(
    symbol='IBM',
    max_position_size=100,
    stop_loss_pct=0.05  # 5% stop loss
)
```

### Data Analysis and Backtesting

```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TradingAnalyzer:
    def __init__(self, csv_file='IBM.csv'):
        self.data = pd.read_csv(csv_file)
        self.ai_model = PortfolioManagementModel()
    
    def analyze_model_performance(self):
        """Analyze AI model performance on historical data"""
        X = self.data[['Delta Close']]
        y_true = self.data['Signal']
        
        # Get predictions
        predictions = self.ai_model.network.predict(X.values)
        y_pred = np.around(predictions.flatten())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1-Score': f1_score(y_true, y_pred, average='weighted')
        }
        
        print("Model Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def backtest_strategy(self, initial_capital=10000):
        """Backtest the trading strategy"""
        capital = initial_capital
        position = 0
        trades = []
        portfolio_values = [capital]
        
        for i, row in self.data.iterrows():
            delta = row['Delta Close']
            
            # Get AI prediction
            prediction = self.ai_model.network.predict([[delta]])
            decision = np.around(prediction[0][0])
            
            # Simulate price (for backtesting purposes)
            price = 100 + delta  # Simplified price simulation
            
            # Execute trades based on AI decision
            if decision >= 0.5 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    capital += position * price
                    position = 0
                
                shares_to_buy = int(capital * 0.95 / price)  # Use 95% of capital
                if shares_to_buy > 0:
                    capital -= shares_to_buy * price
                    position += shares_to_buy
                    trades.append({
                        'Date': i,
                        'Action': 'BUY',
                        'Shares': shares_to_buy,
                        'Price': price,
                        'Capital': capital
                    })
            
            elif decision <= -0.5 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    capital += position * price
                    trades.append({
                        'Date': i,
                        'Action': 'SELL',
                        'Shares': position,
                        'Price': price,
                        'Capital': capital
                    })
                    position = 0
            
            # Calculate portfolio value
            portfolio_value = capital + (position * price if position > 0 else 0)
            portfolio_values.append(portfolio_value)
        
        # Final portfolio value
        final_price = 100 + self.data['Delta Close'].iloc[-1]
        final_value = capital + (position * final_price if position > 0 else 0)
        
        # Calculate returns
        total_return = (final_value - initial_capital) / initial_capital
        
        print(f"\nBacktest Results:")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Number of Trades: {len(trades)}")
        
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'total_return': total_return,
            'final_value': final_value
        }
    
    def plot_performance(self, backtest_results):
        """Plot backtest performance"""
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value over time
        plt.subplot(2, 1, 1)
        plt.plot(backtest_results['portfolio_values'])
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Plot price deltas
        plt.subplot(2, 1, 2)
        plt.plot(self.data['Delta Close'])
        plt.title('Price Deltas')
        plt.ylabel('Price Change')
        plt.xlabel('Time Period')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Usage
analyzer = TradingAnalyzer()
metrics = analyzer.analyze_model_performance()
backtest_results = analyzer.backtest_strategy()
analyzer.plot_performance(backtest_results)
```

## Customization Examples

### Custom Model Architecture

```python
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
import pandas as pd
import numpy as np

class CustomModelDevelopment:
    def __init__(self, model_type='lstm'):
        self.model_type = model_type
        
    def create_lstm_model(self, input_shape):
        """Create LSTM model for time series prediction"""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1, activation='tanh'))
        return model
    
    def create_deep_model(self, input_shape):
        """Create deeper neural network"""
        model = Sequential()
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        return model
    
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_custom_model(self):
        # Load data
        data = pd.read_csv('IBM.csv')
        
        if self.model_type == 'lstm':
            # Prepare LSTM data
            prices = data['Delta Close'].values.reshape(-1, 1)
            X, y = self.prepare_lstm_data(prices)
            model = self.create_lstm_model((X.shape[1], 1))
        else:
            # Prepare standard data
            X = data[['Delta Close']].values
            y = data['Signal'].values
            model = self.create_deep_model((1,))
        
        # Compile and train
        model.compile(
            optimizer='adam',
            loss='mse' if self.model_type == 'lstm' else 'hinge',
            metrics=['accuracy']
        )
        
        history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
        
        # Save model
        model_json = model.to_json()
        with open(f"custom_{self.model_type}_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"custom_{self.model_type}_weights.h5")
        
        print(f"Custom {self.model_type} model trained and saved!")
        return model, history

# Usage
custom_dev = CustomModelDevelopment('deep')
model, history = custom_dev.train_custom_model()
```

### Environment Configuration

```python
import os
import json
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    # API Configuration
    alpaca_key_id: str
    alpaca_secret_key: str
    alpaca_base_url: str
    
    # Trading Parameters
    symbol: str
    time_frame: int
    max_position_size: int
    stop_loss_pct: float
    
    # Model Configuration
    model_file: str
    weights_file: str
    data_file: str
    
    # Risk Management
    risk_per_trade: float
    max_daily_trades: int
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            alpaca_key_id=os.getenv('ALPACA_KEY_ID', 'PKPO0ZH3XTVB336B7TEO'),
            alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY', 'your_secret_key'),
            alpaca_base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            symbol=os.getenv('TRADING_SYMBOL', 'IBM'),
            time_frame=int(os.getenv('TIME_FRAME', '86400')),
            max_position_size=int(os.getenv('MAX_POSITION_SIZE', '100')),
            stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', '0.05')),
            model_file=os.getenv('MODEL_FILE', 'model.json'),
            weights_file=os.getenv('WEIGHTS_FILE', 'weights.h5'),
            data_file=os.getenv('DATA_FILE', 'IBM.csv'),
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', '10'))
        )
    
    @classmethod
    def from_file(cls, config_file='config.json'):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def save_to_file(self, config_file='config.json'):
        """Save configuration to JSON file"""
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

# Usage
config = TradingConfig.from_env()
config.save_to_file()

# Or create custom configuration
custom_config = TradingConfig(
    alpaca_key_id='your_key',
    alpaca_secret_key='your_secret',
    alpaca_base_url='https://paper-api.alpaca.markets',
    symbol='AAPL',
    time_frame=3600,
    max_position_size=50,
    stop_loss_pct=0.03,
    model_file='aapl_model.json',
    weights_file='aapl_weights.h5',
    data_file='AAPL.csv',
    risk_per_trade=0.015,
    max_daily_trades=5
)
```

## Best Practices

### 1. Data Management

```python
import pandas as pd
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, symbol='IBM'):
        self.symbol = symbol
    
    def validate_data_quality(self, df):
        """Validate data quality before training"""
        issues = []
        
        # Check for missing values
        if df.isnull().sum().any():
            issues.append("Missing values detected")
        
        # Check for outliers
        Q1 = df['Delta Close'].quantile(0.25)
        Q3 = df['Delta Close'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['Delta Close'] < Q1 - 1.5*IQR) | 
                      (df['Delta Close'] > Q3 + 1.5*IQR)]
        
        if len(outliers) > len(df) * 0.1:  # More than 10% outliers
            issues.append(f"High number of outliers: {len(outliers)}")
        
        # Check signal distribution
        signal_dist = df['Signal'].value_counts()
        if signal_dist.min() < len(df) * 0.1:  # Less than 10% of any signal
            issues.append("Imbalanced signal distribution")
        
        return issues
    
    def clean_data(self, df):
        """Clean and preprocess data"""
        # Remove missing values
        df = df.dropna()
        
        # Cap extreme outliers
        Q1 = df['Delta Close'].quantile(0.01)
        Q99 = df['Delta Close'].quantile(0.99)
        df['Delta Close'] = df['Delta Close'].clip(Q1, Q99)
        
        # Ensure signals are in correct format
        df['Signal'] = df['Signal'].astype(int)
        df['Signal'] = df['Signal'].clip(-1, 1)
        
        return df
    
    def create_features(self, df):
        """Create additional features for better predictions"""
        # Moving averages
        df['MA_5'] = df['Delta Close'].rolling(window=5).mean()
        df['MA_20'] = df['Delta Close'].rolling(window=20).mean()
        
        # Volatility
        df['Volatility'] = df['Delta Close'].rolling(window=10).std()
        
        # Momentum
        df['Momentum'] = df['Delta Close'].diff(5)
        
        # Remove NaN values created by rolling operations
        df = df.dropna()
        
        return df

# Usage
data_manager = DataManager()
data = pd.read_csv('IBM.csv')
issues = data_manager.validate_data_quality(data)
if issues:
    print("Data quality issues:", issues)
    data = data_manager.clean_data(data)
data = data_manager.create_features(data)
```

### 2. Error Handling and Logging

```python
import logging
import traceback
from functools import wraps

def setup_comprehensive_logging():
    """Set up comprehensive logging system"""
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    file_handler = logging.FileHandler('trading_system_detailed.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    error_handler = logging.FileHandler('trading_errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler, error_handler]
    )

def handle_exceptions(func):
    """Decorator for handling exceptions with logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

class RobustTradingSystem(TradingSystem):
    def __init__(self, *args, **kwargs):
        setup_comprehensive_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(*args, **kwargs)
    
    @handle_exceptions
    def place_buy_order(self):
        self.logger.info(f"Attempting to place buy order for {self.symbol}")
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=1,
                side='buy',
                type='market',
                time_in_force='day'
            )
            self.logger.info(f"Buy order successful: {order.id}")
            return order
        except Exception as e:
            self.logger.error(f"Buy order failed: {e}")
            raise
    
    @handle_exceptions
    def place_sell_order(self):
        self.logger.info(f"Attempting to place sell order for {self.symbol}")
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=1,
                side='sell',
                type='market',
                time_in_force='day'
            )
            self.logger.info(f"Sell order successful: {order.id}")
            return order
        except Exception as e:
            self.logger.error(f"Sell order failed: {e}")
            raise
    
    def system_loop(self):
        self.logger.info("Starting robust trading system loop")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                # Your trading logic here
                time.sleep(self.time_frame)
                consecutive_errors = 0  # Reset error count on success
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"System loop error #{consecutive_errors}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical("Max consecutive errors reached. Stopping system.")
                    break
                
                # Exponential backoff
                sleep_time = min(300, 30 * (2 ** consecutive_errors))
                self.logger.info(f"Waiting {sleep_time} seconds before retry")
                time.sleep(sleep_time)
```

### 3. Performance Monitoring

```python
import time
import psutil
import threading
from collections import defaultdict, deque

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.trade_history = deque(maxlen=1000)  # Keep last 1000 trades
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            self.metrics['cpu'].append(cpu_percent)
            self.metrics['memory'].append(memory_percent)
            
            # Keep only last 100 measurements
            if len(self.metrics['cpu']) > 100:
                self.metrics['cpu'].pop(0)
                self.metrics['memory'].pop(0)
            
            time.sleep(60)  # Monitor every minute
    
    def log_trade(self, action, symbol, quantity, price, timestamp=None):
        """Log a trade for performance analysis"""
        if timestamp is None:
            timestamp = time.time()
        
        trade = {
            'timestamp': timestamp,
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price
        }
        
        self.trade_history.append(trade)
    
    def get_performance_report(self):
        """Generate performance report"""
        uptime = time.time() - self.start_time
        
        report = {
            'uptime_hours': uptime / 3600,
            'total_trades': len(self.trade_history),
            'trades_per_hour': len(self.trade_history) / (uptime / 3600) if uptime > 0 else 0,
            'avg_cpu': sum(self.metrics['cpu']) / len(self.metrics['cpu']) if self.metrics['cpu'] else 0,
            'avg_memory': sum(self.metrics['memory']) / len(self.metrics['memory']) if self.metrics['memory'] else 0,
            'recent_trades': list(self.trade_history)[-10:]  # Last 10 trades
        }
        
        return report
    
    def print_report(self):
        """Print formatted performance report"""
        report = self.get_performance_report()
        
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        print(f"System Uptime: {report['uptime_hours']:.2f} hours")
        print(f"Total Trades: {report['total_trades']}")
        print(f"Trades per Hour: {report['trades_per_hour']:.2f}")
        print(f"Average CPU Usage: {report['avg_cpu']:.1f}%")
        print(f"Average Memory Usage: {report['avg_memory']:.1f}%")
        
        if report['recent_trades']:
            print("\nRecent Trades:")
            for trade in report['recent_trades']:
                timestamp = datetime.fromtimestamp(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {timestamp}: {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        print("="*50)

# Usage with trading system
monitor = PerformanceMonitor()
monitor.start_monitoring()

# In your trading system, log trades:
# monitor.log_trade('BUY', 'IBM', 10, 150.25)

# Generate reports periodically:
# monitor.print_report()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Files Not Found
```python
import os

def check_model_files():
    required_files = ['model.json', 'weights.h5', 'IBM.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        print("Solution: Run PMModelDevelopment() to create model files")
        return False
    return True

# Check before starting system
if check_model_files():
    trading_system = PortfolioManagementSystem()
else:
    print("Creating model files first...")
    model_dev = PMModelDevelopment()
    trading_system = PortfolioManagementSystem()
```

#### 2. API Connection Issues
```python
def test_api_connection():
    try:
        api = AlpacaPaperSocket()
        account = api.get_account()
        print(f"✓ API connection successful")
        print(f"Account: {account.id}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        return True
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        print("Solutions:")
        print("1. Check your internet connection")
        print("2. Verify API credentials")
        print("3. Check Alpaca service status")
        return False

# Test connection before trading
if test_api_connection():
    print("Proceeding with trading system...")
else:
    print("Please fix API connection issues first")
```

#### 3. Memory Issues
```python
import gc
import psutil

def monitor_memory_usage():
    """Monitor and manage memory usage"""
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    
    if memory.percent > 80:
        print("High memory usage detected. Running garbage collection...")
        gc.collect()
        
        # Clear unnecessary variables
        import sys
        for name in list(sys.modules.keys()):
            if name.startswith('temp_') or name.startswith('cache_'):
                del sys.modules[name]
        
        memory_after = psutil.virtual_memory()
        print(f"Memory usage after cleanup: {memory_after.percent}%")

# Use in system loop
def system_loop(self):
    while True:
        try:
            # Your trading logic
            
            # Monitor memory every 10 iterations
            if hasattr(self, 'loop_count'):
                self.loop_count += 1
            else:
                self.loop_count = 1
            
            if self.loop_count % 10 == 0:
                monitor_memory_usage()
            
            time.sleep(self.time_frame)
        except Exception as e:
            print(f"Error: {e}")
```

## Performance Optimization

### 1. Efficient Data Handling

```python
import numpy as np
import pandas as pd
from functools import lru_cache

class OptimizedDataHandler:
    def __init__(self):
        self.data_cache = {}
    
    @lru_cache(maxsize=100)
    def get_price_data(self, symbol, timeframe, limit):
        """Cached price data retrieval"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        if cache_key in self.data_cache:
            cache_time, data = self.data_cache[cache_key]
            if time.time() - cache_time < 60:  # Cache for 1 minute
                return data
        
        # Fetch new data
        api = AlpacaPaperSocket()
        bars = api.get_barset(symbol, timeframe, limit)
        
        # Cache the result
        self.data_cache[cache_key] = (time.time(), bars)
        return bars
    
    def batch_predictions(self, deltas, batch_size=32):
        """Process predictions in batches for efficiency"""
        ai_model = PortfolioManagementModel()
        
        predictions = []
        for i in range(0, len(deltas), batch_size):
            batch = deltas[i:i+batch_size]
            batch_array = np.array(batch).reshape(-1, 1)
            batch_predictions = ai_model.network.predict(batch_array)
            predictions.extend(batch_predictions.flatten())
        
        return predictions
```

### 2. Asynchronous Operations

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncTradingSystem(TradingSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def async_get_price_data(self, symbol):
        """Asynchronously get price data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.api.get_barset,
            symbol, '1H', 2
        )
    
    async def async_place_order(self, side, symbol, qty):
        """Asynchronously place order"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.api.submit_order,
            symbol, qty, side, 'market', 'day'
        )
    
    async def async_system_loop(self):
        """Asynchronous system loop"""
        while True:
            try:
                # Get price data asynchronously
                bars = await self.async_get_price_data(self.symbol)
                
                if len(bars[self.symbol]) >= 2:
                    current = bars[self.symbol][-1].c
                    previous = bars[self.symbol][-2].c
                    delta = current - previous
                    
                    # Make prediction
                    prediction = self.ai_model.network.predict([[delta]])
                    decision = np.around(prediction[0][0])
                    
                    # Place orders asynchronously
                    if decision >= 0.5:
                        await self.async_place_order('buy', self.symbol, 1)
                    elif decision <= -0.5:
                        await self.async_place_order('sell', self.symbol, 1)
                
                await asyncio.sleep(self.time_frame)
                
            except Exception as e:
                print(f"Async system loop error: {e}")
                await asyncio.sleep(60)
    
    def system_loop(self):
        """Override system_loop to use async version"""
        asyncio.run(self.async_system_loop())
```

This comprehensive documentation provides everything needed to understand, use, and customize the AI Stock Trading system. Each section includes practical examples, best practices, and solutions to common issues.