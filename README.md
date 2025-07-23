# AI Stock Trading System

A comprehensive AI-powered stock trading system that uses machine learning to make automated trading decisions. The system integrates with Alpaca's paper trading API and uses neural networks for prediction-based trading.

## 📚 Documentation

This project includes comprehensive documentation covering all aspects of the system:

### Core Documentation
- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference with examples and usage instructions
- **[Function Reference](FUNCTION_REFERENCE.md)** - Detailed documentation of all functions, methods, and parameters
- **[Usage Guide](USAGE_GUIDE.md)** - Step-by-step tutorials, examples, and best practices
- **[Component Guide](COMPONENT_GUIDE.md)** - System architecture, design patterns, and component relationships

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the AI Model**:
   ```python
   from AI_Stock_Trading import PMModelDevelopment
   model_dev = PMModelDevelopment()
   ```

3. **Start Trading**:
   ```python
   from AI_Stock_Trading import PortfolioManagementSystem
   trading_system = PortfolioManagementSystem()
   ```

## 🏗️ System Architecture

The system is built with a modular architecture consisting of:

- **AlpacaPaperSocket**: API connection and authentication
- **TradingSystem**: Abstract base class for trading strategies  
- **PMModelDevelopment**: AI model training and development
- **PortfolioManagementModel**: Model loading and prediction
- **PortfolioManagementSystem**: Complete trading implementation

## 🔧 Key Features

- **AI-Powered Decisions**: Neural network-based trading predictions
- **Paper Trading**: Safe testing with Alpaca's paper trading API
- **Modular Design**: Easy to extend and customize
- **Background Processing**: Automated trading in separate threads
- **Comprehensive Testing**: Full test suite with mocks and integration tests
- **Risk Management**: Built-in safeguards and error handling

## 📊 Data Requirements

The system requires historical data in CSV format with columns:
- `Delta Close`: Price change values
- `Signal`: Trading signals (-1 for sell, 0 for hold, 1 for buy)

## 🚀 Getting Started

For detailed setup instructions, examples, and customization options, see the [Usage Guide](USAGE_GUIDE.md).

## 🧪 Testing

Run the test suite:
```bash
cd tests
python -m pytest test_trading_system.py -v
```

## 📈 Performance

The system is designed for:
- Real-time trading decisions
- Low latency predictions
- Efficient memory usage
- Robust error handling

## 🔒 Security

- Uses paper trading by default for safety
- API credentials should be stored as environment variables
- Includes comprehensive error handling and logging

## 📝 License

This project is designed as a educational framework for AI trading system development.
