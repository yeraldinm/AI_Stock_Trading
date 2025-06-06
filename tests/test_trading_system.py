import sys
import types
from unittest.mock import MagicMock, patch
import pathlib


def load_trading_module():
    """Load the AI_Stock_Trading module without executing the demo code."""
    # Stub external modules to avoid ImportError
    for name in ['pandas', 'numpy']:
        sys.modules.setdefault(name, types.ModuleType(name))

    # Minimal keras stub
    keras = types.ModuleType('keras')
    keras.layers = types.ModuleType('keras.layers')
    keras.layers.Dense = MagicMock()
    keras.models = types.ModuleType('keras.models')
    keras.models.Sequential = MagicMock(return_value=MagicMock())
    keras.models.model_from_json = MagicMock(return_value=MagicMock())
    sys.modules['keras'] = keras
    sys.modules['keras.layers'] = keras.layers
    sys.modules['keras.models'] = keras.models

    # Minimal sklearn stub
    sklearn = types.ModuleType('sklearn')
    sklearn_ms = types.ModuleType('sklearn.model_selection')
    sklearn_ms.train_test_split = MagicMock(return_value=(None, None, None, None))
    sklearn_metrics = types.ModuleType('sklearn.metrics')
    sklearn_metrics.classification_report = MagicMock(return_value=None)
    sklearn.model_selection = sklearn_ms
    sklearn.metrics = sklearn_metrics
    sys.modules['sklearn'] = sklearn
    sys.modules['sklearn.model_selection'] = sklearn_ms
    sys.modules['sklearn.metrics'] = sklearn_metrics

    # Mock Alpaca REST
    alpaca = types.ModuleType('alpaca_trade_api')
    alpaca.REST = MagicMock()
    sys.modules['alpaca_trade_api'] = alpaca

    path = pathlib.Path(__file__).resolve().parents[1] / 'AI_Stock_Trading.py'
    lines = path.read_text().splitlines()
    filtered = []
    for line in lines:
        if line.strip().startswith('PortfolioManagementModel()') or line.strip().startswith('PortfolioManagementSystem()'):
            break
        filtered.append(line)
    source = '\n'.join(filtered)
    module = types.ModuleType('AI_Stock_Trading')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


trading_module = load_trading_module()
TradingSystem = trading_module.TradingSystem


class DummySystem(TradingSystem):
    def place_buy_order(self):
        pass

    def place_sell_order(self):
        pass

    def system_loop(self):
        pass


def test_thread_started():
    thread_instance = MagicMock()
    with patch('threading.Thread', return_value=thread_instance) as thread_cls:
        system = DummySystem(api=MagicMock(), symbol='SYM', time_frame=1, system_id=1, system_label='test')
        thread_cls.assert_called_once()
        target = thread_cls.call_args.kwargs['target']
        assert target.__self__ is system
        assert target.__func__ is system.system_loop.__func__
        thread_instance.start.assert_called_once()
