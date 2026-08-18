"""Reflection pass for COC v3.

Post-generation single-shot reflection: takes a draft answer plus the
same evidence the reasoner already saw and looks for unsupported
claims, contradictions with the evidence, or missed evidence. If a
problem is found it revises the draft once and records the outcome on
``EngineResult`` via a ``ReflectionReport``.

This package intentionally does NOT depend on goal internals,
self-consistency internals, memory, learning, adaptation, or any
scientific-discovery machinery. It is a thin, deterministic layer that
sits between generation and the final result.
"""

from reflection.contracts import ReflectionReport
from reflection.reflector import Reflector

__all__ = ["ReflectionReport", "Reflector"]
