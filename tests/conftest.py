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
from cpt_mem0.planner import (  # noqa: E402
    Hardware,
    PlannerConfig,
    plan_for_hardware,
)


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


@pytest.fixture
def candidate(catalogue):
    """A planned candidate for the 12 GB reference card.

    Used by the tests that check the launch flags and the KV semantics
    against a real configuration.
    """
    hardware = Hardware(total_vram_mib=12288, name="RTX 3060",
                        vendor="nvidia")
    result = plan_for_hardware(hardware, catalogue.sorted_entries(),
                               PlannerConfig())
    assert result.viable is True
    assert result.candidate is not None
    return result.candidate
