# Observability package initialization
from .decorators import span
from .tracer import tracer

__all__ = ["span", "tracer"]