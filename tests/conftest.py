"""Shared fixtures (pytest finds them here without an import)."""
import pytest

from distill import data
from helpers import SOURCE


@pytest.fixture
def source(monkeypatch):
    """A Thai set's source questions: SOURCE, of which p0-p3 were used in English."""
    monkeypatch.setattr(data, "load_prompts", lambda cfg, name: SOURCE)
    monkeypatch.setattr(data, "load_complexity", lambda ds: {f"p{i}": 2 for i in range(4)})
