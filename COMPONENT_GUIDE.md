# Component Architecture Guide

## Table of Contents
- [System Overview](#system-overview)
- [Architecture Patterns](#architecture-patterns)
- [Component Relationships](#component-relationships)
- [Data Flow](#data-flow)
- [Design Principles](#design-principles)
- [Extension Points](#extension-points)
- [Testing Strategy](#testing-strategy)

## System Overview

The AI Stock Trading System is built using a modular, object-oriented architecture that separates concerns and provides clear interfaces for extension and customization.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Stock Trading System                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   Data Layer    │  │  Model Layer    │  │ Trading Layer   ││
│  │                 │  │                 │  │                 ││
│  │ • CSV Files     │  │ • AI Models     │  │ • Trading       ││
│  │ • Market Data   │  │ • Predictions   │  │   Systems       ││
│  │ • Price Feeds   │  │ • Training      │  │ • Order         ││
│  │                 │  │                 │  │   Execution     ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │  API Layer      │  │ Threading Layer │  │  Utilities      ││
│  │                 │  │                 │  │                 ││
│  │ • Alpaca API    │  │ • Background    │  │ • Logging       ││
│  │ • REST Client   │  │   Threads       │  │ • Error         ││
│  │ • Authentication│  │ • Async Ops     │  │   Handling      ││
│  │                 │  │                 │  │ • Monitoring    ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **AlpacaPaperSocket**: API connection and authentication
2. **TradingSystem**: Abstract base class for trading strategies
3. **PMModelDevelopment**: AI model training and development
4. **PortfolioManagementModel**: Model loading and prediction
5. **PortfolioManagementSystem**: Complete trading implementation

## Architecture Patterns

### 1. Abstract Factory Pattern

The `TradingSystem` abstract base class implements the Abstract Factory pattern, providing a common interface for creating different types of trading systems.

```python
# Abstract Factory
class TradingSystem(abc.ABC):
    def __init__(self, api, symbol, time_frame, system_id, system_label):
        # Common initialization
        pass
    
    @abc.abstractmethod
    def place_buy_order(self):
        pass
    
    @abc.abstractmethod
    def place_sell_order(self):
        pass
    
    @abc.abstractmethod
    def system_loop(self):
        pass

# Concrete Implementations
class PortfolioManagementSystem(TradingSystem):
    # Specific implementation for AI-based trading
    pass

class CustomTradingSystem(TradingSystem):
    # Custom implementation with different strategy
    pass
```

**Benefits**:
- Consistent interface across all trading systems
- Easy to add new trading strategies
- Polymorphic behavior for different implementations

### 2. Strategy Pattern

The AI model acts as a strategy that can be swapped out for different prediction algorithms.

```python
class TradingStrategy:
    def __init__(self, prediction_strategy):
        self.prediction_strategy = prediction_strategy
    
    def make_decision(self, market_data):
        return self.prediction_strategy.predict(market_data)

# Different strategies
class NeuralNetworkStrategy:
    def predict(self, data):
        # Neural network prediction logic
        pass

class RandomForestStrategy:
    def predict(self, data):
        # Random forest prediction logic
        pass
```

### 3. Observer Pattern

The system uses threading to implement an observer-like pattern where the main thread continues while trading operations happen in the background.

```python
class TradingSystem:
    def __init__(self, ...):
        # Start background observer thread
        thread = threading.Thread(target=self.system_loop)
        thread.start()
    
    def system_loop(self):
        # Continuously observe market conditions
        while True:
            self.observe_market()
            self.make_trading_decision()
            time.sleep(self.time_frame)
```

### 4. Template Method Pattern

The `TradingSystem` class uses the Template Method pattern to define the skeleton of the trading algorithm.

```python
class TradingSystem:
    def execute_trading_cycle(self):
        # Template method defining the algorithm structure
        market_data = self.get_market_data()
        decision = self.analyze_data(market_data)
        if decision == 'BUY':
            self.place_buy_order()
        elif decision == 'SELL':
            self.place_sell_order()
    
    # Abstract methods to be implemented by subclasses
    @abc.abstractmethod
    def place_buy_order(self):
        pass
    
    @abc.abstractmethod
    def place_sell_order(self):
        pass
```

### 5. Facade Pattern

The `AlpacaPaperSocket` class acts as a facade, simplifying the interface to the complex Alpaca API.

```python
class AlpacaPaperSocket(REST):
    def __init__(self):
        # Simplify API initialization
        super().__init__(
            key_id='predefined_key',
            secret_key='predefined_secret',
            base_url='https://paper-api.alpaca.markets'
        )
    
    # Inherited methods provide simplified interface
    # to complex trading operations
```

## Component Relationships

### Dependency Graph

```
AlpacaPaperSocket
    ↑
    │
TradingSystem (Abstract)
    ↑
    │
PortfolioManagementSystem
    ↑
    │
PortfolioManagementModel ← PMModelDevelopment
    ↑
    │
Neural Network (Keras)
```

### Interaction Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PMModelDev      │    │ PortfolioMgmt   │    │ PortfolioMgmt   │
│ Development     │───▶│ Model           │───▶│ System          │
│                 │    │                 │    │                 │
│ • Train Model   │    │ • Load Model    │    │ • Use Model     │
│ • Save Files    │    │ • Make Predict  │    │ • Execute       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ model.json      │    │ Keras Network   │    │ AlpacaPaper     │
│ weights.h5      │    │ Predictions     │    │ Socket          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Communication

1. **Model Development → Model Loading**:
   ```python
   # PMModelDevelopment creates files
   PMModelDevelopment() # Creates model.json, weights.h5
   
   # PortfolioManagementModel loads files
   model = PortfolioManagementModel() # Loads from files
   ```

2. **Model → Trading System**:
   ```python
   # Trading system uses model for predictions
   class PortfolioManagementSystem(TradingSystem):
       def __init__(self):
           self.AI = PortfolioManagementModel()
       
       def make_decision(self, delta):
           return self.AI.network.predict([[delta]])
   ```

3. **Trading System → API**:
   ```python
   # Trading system uses API for execution
   def place_buy_order(self):
       self.api.submit_order(...)
   ```

## Data Flow

### Training Phase Data Flow

```
IBM.csv
   │
   ▼
┌─────────────────┐
│ Load CSV Data   │
│ • Delta Close   │
│ • Signal        │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│ Data Preprocessing│
│ • Train/Test    │
│ • Split         │
│ • Validation    │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│ Neural Network  │
│ • 5 Layers      │
│ • Tanh Activation│
│ • RMSprop Opt   │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│ Model Files     │
│ • model.json    │
│ • weights.h5    │
└─────────────────┘
```

### Trading Phase Data Flow

```
Market Data (Alpaca API)
   │
   ▼
┌─────────────────┐
│ Price Collection│
│ • Daily Bars    │
│ • Close Prices  │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│ Delta Calculation│
│ • Weekly Change │
│ • Price Delta   │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│ AI Prediction   │
│ • Load Model    │
│ • Predict       │
│ • Round Result  │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│ Trading Decision│
│ • Buy (≥ 0.5)   │
│ • Sell (≤ -0.5) │
│ • Hold (else)   │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│ Order Execution │
│ • Market Orders │
│ • Alpaca API    │
└─────────────────┘
```

### State Management

The system maintains several types of state:

1. **Model State**:
   ```python
   # Persistent state in files
   model.json      # Network architecture
   weights.h5      # Trained parameters
   
   # Runtime state in memory
   self.network    # Loaded Keras model
   ```

2. **Trading State**:
   ```python
   # System configuration
   self.symbol = 'IBM'
   self.time_frame = 86400
   self.system_id = 1
   
   # Runtime trading state
   this_weeks_close = 0
   last_weeks_close = 0
   day_count = 0
   ```

3. **API State**:
   ```python
   # Connection state
   self.api = AlpacaPaperSocket()
   
   # Account state (retrieved dynamically)
   account = self.api.get_account()
   positions = self.api.list_positions()
   ```

## Design Principles

### 1. Separation of Concerns

Each component has a single, well-defined responsibility:

- **AlpacaPaperSocket**: API communication only
- **PMModelDevelopment**: Model training only
- **PortfolioManagementModel**: Model loading and prediction only
- **PortfolioManagementSystem**: Trading logic coordination only

### 2. Single Responsibility Principle

```python
# Good: Each class has one reason to change
class DataLoader:
    def load_csv(self, filename):
        pass

class ModelTrainer:
    def train_model(self, data):
        pass

class ModelSaver:
    def save_model(self, model, filename):
        pass

# Bad: Multiple responsibilities in one class
class ModelManager:
    def load_csv(self, filename):
        pass
    def train_model(self, data):
        pass
    def save_model(self, model):
        pass
    def make_predictions(self, data):
        pass
```

### 3. Open/Closed Principle

The system is open for extension but closed for modification:

```python
# Base class is closed for modification
class TradingSystem(abc.ABC):
    # Stable interface
    pass

# But open for extension
class CustomStrategy(TradingSystem):
    def place_buy_order(self):
        # Custom implementation
        pass
```

### 4. Dependency Inversion

High-level modules don't depend on low-level modules:

```python
# High-level trading system
class TradingSystem:
    def __init__(self, api):  # Depends on abstraction
        self.api = api
    
    def execute_trade(self):
        self.api.submit_order(...)  # Uses interface

# Low-level implementation
class AlpacaPaperSocket(REST):
    def submit_order(self, ...):  # Implements interface
        pass
```

### 5. Composition Over Inheritance

The system favors composition:

```python
class PortfolioManagementSystem(TradingSystem):
    def __init__(self):
        # Composition: has-a relationship
        self.AI = PortfolioManagementModel()
        self.api = AlpacaPaperSocket()
        
        # Rather than deep inheritance hierarchies
```

## Extension Points

### 1. Custom Trading Strategies

Extend the `TradingSystem` base class:

```python
class MomentumTradingSystem(TradingSystem):
    def __init__(self, momentum_period=14):
        super().__init__(...)
        self.momentum_period = momentum_period
    
    def calculate_momentum(self, prices):
        return prices[-1] - prices[-self.momentum_period]
    
    def system_loop(self):
        while True:
            prices = self.get_price_history()
            momentum = self.calculate_momentum(prices)
            
            if momentum > threshold:
                self.place_buy_order()
            elif momentum < -threshold:
                self.place_sell_order()
```

### 2. Custom AI Models

Replace the neural network with different algorithms:

```python
class RandomForestModel:
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()
        self.load_model()
    
    def predict(self, features):
        return self.model.predict(features)

class CustomPortfolioSystem(TradingSystem):
    def __init__(self):
        super().__init__(...)
        self.AI = RandomForestModel()  # Different model
```

### 3. Custom Data Sources

Extend data input capabilities:

```python
class AlternativeDataLoader:
    def load_news_sentiment(self, symbol):
        # Load news sentiment data
        pass
    
    def load_social_media_data(self, symbol):
        # Load social media sentiment
        pass

class EnhancedTradingSystem(TradingSystem):
    def __init__(self):
        super().__init__(...)
        self.data_loader = AlternativeDataLoader()
    
    def get_enhanced_features(self, symbol):
        price_data = self.get_price_data(symbol)
        news_sentiment = self.data_loader.load_news_sentiment(symbol)
        social_data = self.data_loader.load_social_media_data(symbol)
        return combine_features(price_data, news_sentiment, social_data)
```

### 4. Custom Risk Management

Add risk management components:

```python
class RiskManager:
    def __init__(self, max_position_size, stop_loss_pct):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
    
    def check_position_size(self, intended_size):
        return min(intended_size, self.max_position_size)
    
    def check_stop_loss(self, entry_price, current_price, position_type):
        # Risk management logic
        pass

class RiskManagedSystem(TradingSystem):
    def __init__(self):
        super().__init__(...)
        self.risk_manager = RiskManager(100, 0.05)
    
    def place_buy_order(self):
        size = self.calculate_position_size()
        safe_size = self.risk_manager.check_position_size(size)
        # Execute with safe size
```

## Testing Strategy

### 1. Unit Testing

Test individual components in isolation:

```python
import unittest
from unittest.mock import Mock, patch

class TestPortfolioManagementModel(unittest.TestCase):
    def setUp(self):
        # Mock file dependencies
        self.mock_model_file = Mock()
        self.mock_weights_file = Mock()
    
    @patch('builtins.open')
    @patch('keras.models.model_from_json')
    def test_model_loading(self, mock_model_from_json, mock_open):
        # Test model loading functionality
        model = PortfolioManagementModel()
        self.assertIsNotNone(model.network)
    
    def test_prediction_format(self):
        model = PortfolioManagementModel()
        prediction = model.network.predict([[1.5]])
        self.assertIsInstance(prediction, np.ndarray)
```

### 2. Integration Testing

Test component interactions:

```python
class TestTradingSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_api = Mock()
        self.trading_system = PortfolioManagementSystem()
        self.trading_system.api = self.mock_api
    
    def test_buy_order_execution(self):
        # Test that AI prediction leads to correct order
        with patch.object(self.trading_system.AI, 'network') as mock_network:
            mock_network.predict.return_value = [[0.8]]  # Strong buy signal
            
            # Trigger decision making
            self.trading_system.make_trading_decision(2.5)
            
            # Verify order was placed
            self.mock_api.submit_order.assert_called_with(
                symbol='IBM',
                qty=1,
                side='buy',
                type='market',
                time_in_force='day'
            )
```

### 3. Mock Testing

Use mocks to test without external dependencies:

```python
class TestWithMocks(unittest.TestCase):
    @patch('AI_Stock_Trading.AlpacaPaperSocket')
    @patch('AI_Stock_Trading.PortfolioManagementModel')
    def test_system_initialization(self, mock_model, mock_api):
        # Test system initialization without real API or model
        system = PortfolioManagementSystem()
        
        # Verify mocks were called
        mock_api.assert_called_once()
        mock_model.assert_called_once()
```

### 4. Behavioral Testing

Test system behavior under different market conditions:

```python
class TestMarketScenarios(unittest.TestCase):
    def test_bull_market_behavior(self):
        # Simulate bull market conditions
        price_deltas = [1.5, 2.0, 1.8, 2.2, 1.9]  # Positive deltas
        
        system = PortfolioManagementSystem()
        decisions = []
        
        for delta in price_deltas:
            prediction = system.AI.network.predict([[delta]])
            decisions.append(np.around(prediction[0][0]))
        
        # Expect mostly buy decisions in bull market
        buy_decisions = sum(1 for d in decisions if d >= 0.5)
        self.assertGreater(buy_decisions, len(decisions) / 2)
    
    def test_bear_market_behavior(self):
        # Simulate bear market conditions
        price_deltas = [-1.5, -2.0, -1.8, -2.2, -1.9]  # Negative deltas
        
        system = PortfolioManagementSystem()
        decisions = []
        
        for delta in price_deltas:
            prediction = system.AI.network.predict([[delta]])
            decisions.append(np.around(prediction[0][0]))
        
        # Expect mostly sell decisions in bear market
        sell_decisions = sum(1 for d in decisions if d <= -0.5)
        self.assertGreater(sell_decisions, len(decisions) / 2)
```

### 5. Performance Testing

Test system performance under load:

```python
import time
import threading

class TestPerformance(unittest.TestCase):
    def test_prediction_speed(self):
        model = PortfolioManagementModel()
        
        start_time = time.time()
        for _ in range(1000):
            model.network.predict([[1.0]])
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 1000
        self.assertLess(avg_time, 0.01)  # Less than 10ms per prediction
    
    def test_concurrent_operations(self):
        system = PortfolioManagementSystem()
        results = []
        
        def make_prediction():
            prediction = system.AI.network.predict([[1.5]])
            results.append(prediction)
        
        # Run multiple predictions concurrently
        threads = [threading.Thread(target=make_prediction) for _ in range(10)]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end_time = time.time()
        
        # Verify all predictions completed
        self.assertEqual(len(results), 10)
        # Verify reasonable performance
        self.assertLess(end_time - start_time, 1.0)
```

## Component Lifecycle

### 1. System Initialization

```python
# Phase 1: Model Training (One-time)
model_dev = PMModelDevelopment()  # Creates model files

# Phase 2: Model Loading
ai_model = PortfolioManagementModel()  # Loads model files

# Phase 3: Trading System Start
trading_system = PortfolioManagementSystem()  # Starts background thread

# Phase 4: Continuous Operation
# System runs in background thread indefinitely
```

### 2. Component Dependencies

```python
# Dependency Resolution Order:
1. CSV Data Files (IBM.csv) must exist
2. PMModelDevelopment creates model files
3. PortfolioManagementModel loads model files
4. AlpacaPaperSocket establishes API connection
5. PortfolioManagementSystem coordinates everything
6. Background thread starts trading loop
```

### 3. Error Recovery

```python
class RobustTradingSystem(TradingSystem):
    def system_loop(self):
        while True:
            try:
                # Normal operation
                self.execute_trading_cycle()
            except APIError as e:
                # API-specific error handling
                self.handle_api_error(e)
            except ModelError as e:
                # Model-specific error handling
                self.handle_model_error(e)
            except Exception as e:
                # General error handling
                self.handle_general_error(e)
            finally:
                # Cleanup and recovery
                self.cleanup_resources()
```

This component guide provides a comprehensive understanding of how the AI Stock Trading System is architected, how components interact, and how to extend the system for custom use cases. The modular design makes it easy to understand, test, and modify individual components without affecting the entire system.