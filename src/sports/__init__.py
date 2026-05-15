"""
Sport domain package.

Provides the Sport protocol, SportCapabilities dataclass, and the
REGISTRY mapping every sport key to its capabilities and metadata.

Phase 1 — interface definition only.
Phase 2 — individual sport modules (nba.py, mlb.py, …) that implement
          the full Sport protocol and are registered here.
"""

from src.sports.base import Sport, SportCapabilities
from src.sports.registry import REGISTRY, get_sport, active_sports

__all__ = ["Sport", "SportCapabilities", "REGISTRY", "get_sport", "active_sports"]
