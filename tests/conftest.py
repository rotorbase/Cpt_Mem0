"""Shared test fixtures.

No GPU, no network, no model files. The tests exercise the planner's
arithmetic against the constants measured in docs/verification.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpt_mem0.catalogue import (  # noqa: E402
    Catalogue,
    CatalogueEntry,
    Quantisation,
    catalogue_from_dict,
    load_catalogue,
)
from cpt_mem0.planner import Hardware, PlannerConfig  # noqa: E402


@pytest.fixture(scope="session")
def catalogue() -> Catalogue:
    return load_catalogue(ROOT / "catalogue.json")


@pytest.fixture
def hardware_3060_12gb() -> Hardware:
    return Hardware(total_vram_mib=12288, name="NVIDIA GeForce RTX 3060",
                    vendor="nvidia")


@pytest.fixture
def config() -> PlannerConfig:
    return PlannerConfig()
