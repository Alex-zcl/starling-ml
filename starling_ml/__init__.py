"""Публичный API: конфиг, анализатор, движок; внутренности доступны расширениям."""
from .core.config import load_engine, read_config, save_config
from .core.engine import Engine
from .core.module import Module
from .analysis import analyze_config, AnalysisReport
from .configs import get_config, STANDARD_CONFIGS

__version__ = "0.3.0"
__all__ = ["Engine", "Module", "load_engine", "read_config", "save_config", "analyze_config", "AnalysisReport", "get_config", "STANDARD_CONFIGS"]
