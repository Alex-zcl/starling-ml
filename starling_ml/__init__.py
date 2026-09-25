"""Публичный API: конфиг, анализатор, движок; внутренности доступны расширениям."""
from .core.config import load_engine, read_config, save_config
from .core.engine import Engine
from .core.module import Module
from .analysis import analyze_config, AnalysisReport
from .configs import get_config, get_experiment_config, STANDARD_CONFIGS
from .composition import (
    ConfigConflict,
    ConfigFragment,
    compose,
    fragment,
    module_fragment,
    phase_module,
    recipe,
)
from .monitoring import monitoring_profile

__version__ = "0.4.0"
__all__ = [
    "AnalysisReport",
    "ConfigConflict",
    "ConfigFragment",
    "Engine",
    "Module",
    "STANDARD_CONFIGS",
    "analyze_config",
    "compose",
    "fragment",
    "get_config",
    "get_experiment_config",
    "load_engine",
    "module_fragment",
    "monitoring_profile",
    "phase_module",
    "read_config",
    "recipe",
    "save_config",
]
