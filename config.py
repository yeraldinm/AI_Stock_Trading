"""
Configuration module for the Low Latency Trading Platform
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings
from pydantic import validator
from dotenv import load_dotenv

load_dotenv()


class TradingConfig(BaseSettings):
    """Main configuration class using Pydantic for validation"""
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database Configuration
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "trading_platform")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "trading_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # MongoDB Configuration (for tick data)
    MONGO_HOST: str = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT: int = int(os.getenv("MONGO_PORT", "27017"))
    MONGO_DB: str = os.getenv("MONGO_DB", "market_data")
    MONGO_USER: Optional[str] = os.getenv("MONGO_USER")
    MONGO_PASSWORD: Optional[str] = os.getenv("MONGO_PASSWORD")
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_CONSUMER_GROUP: str = os.getenv("KAFKA_CONSUMER_GROUP", "trading_platform")
    
    # Trading API Configuration
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Risk Management
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "10000.0"))
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "5000.0"))
    MAX_ORDERS_PER_SECOND: int = int(os.getenv("MAX_ORDERS_PER_SECOND", "10"))
    
    # Performance Settings
    ORDER_BOOK_DEPTH: int = int(os.getenv("ORDER_BOOK_DEPTH", "10"))
    TICK_BUFFER_SIZE: int = int(os.getenv("TICK_BUFFER_SIZE", "1000"))
    WEBSOCKET_PING_INTERVAL: int = int(os.getenv("WEBSOCKET_PING_INTERVAL", "30"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL database URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def mongo_url(self) -> str:
        """Get MongoDB URL"""
        if self.MONGO_USER and self.MONGO_PASSWORD:
            return f"mongodb://{self.MONGO_USER}:{self.MONGO_PASSWORD}@{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DB}"
        return f"mongodb://{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DB}"


@dataclass
class InstrumentConfig:
    """Configuration for trading instruments"""
    symbol: str
    exchange: str
    tick_size: float
    lot_size: float
    max_position: float
    margin_requirement: float = 1.0
    enabled: bool = True


@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    name: str
    enabled: bool = True
    max_position: float = 10000.0
    risk_limit: float = 1000.0
    parameters: Dict[str, Any] = field(default_factory=dict)


# Global configuration instance
config = TradingConfig()

# Default instrument configurations
DEFAULT_INSTRUMENTS = [
    InstrumentConfig(
        symbol="AAPL",
        exchange="NASDAQ",
        tick_size=0.01,
        lot_size=1.0,
        max_position=10000.0,
        margin_requirement=0.25
    ),
    InstrumentConfig(
        symbol="MSFT",
        exchange="NASDAQ",
        tick_size=0.01,
        lot_size=1.0,
        max_position=10000.0,
        margin_requirement=0.25
    ),
    InstrumentConfig(
        symbol="GOOGL",
        exchange="NASDAQ",
        tick_size=0.01,
        lot_size=1.0,
        max_position=10000.0,
        margin_requirement=0.25
    ),
    InstrumentConfig(
        symbol="TSLA",
        exchange="NASDAQ",
        tick_size=0.01,
        lot_size=1.0,
        max_position=5000.0,
        margin_requirement=0.30
    ),
    InstrumentConfig(
        symbol="SPY",
        exchange="ARCA",
        tick_size=0.01,
        lot_size=1.0,
        max_position=20000.0,
        margin_requirement=0.20
    )
]

# Default strategy configurations
DEFAULT_STRATEGIES = [
    StrategyConfig(
        name="MeanReversion",
        enabled=True,
        max_position=5000.0,
        risk_limit=500.0,
        parameters={
            "lookback_period": 20,
            "threshold": 2.0,
            "stop_loss": 0.02,
            "take_profit": 0.01
        }
    ),
    StrategyConfig(
        name="MomentumBreakout",
        enabled=True,
        max_position=7500.0,
        risk_limit=750.0,
        parameters={
            "breakout_period": 10,
            "volume_threshold": 1.5,
            "stop_loss": 0.03,
            "take_profit": 0.02
        }
    ),
    StrategyConfig(
        name="AIPredictor",
        enabled=True,
        max_position=10000.0,
        risk_limit=1000.0,
        parameters={
            "model_path": "models/ai_predictor.h5",
            "prediction_threshold": 0.6,
            "rebalance_frequency": 300  # seconds
        }
    )
]