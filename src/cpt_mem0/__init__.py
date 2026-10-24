"""cpt_mem0 - self-contained local mem0 service for consumer GPUs.

The package is importable without a GPU, without network access and without
any model being present on disk. Everything that touches hardware lives
behind an explicit call; the planner is pure arithmetic over measured
constants.
"""

__version__ = "0.1.0"

from .catalogue import (
    Catalogue,
    CatalogueEntry,
    ExtractionMeasurement,
    Quantisation,
)
from .gguf import GGUFError, ModelProfile, read_profile
from .planner import (
    Budget,
    Candidate,
    Hardware,
    PlannerConfig,
    PlanResult,
    plan_for_hardware,
)

__all__ = [
    "__version__",
    "Catalogue",
    "CatalogueEntry",
    "ExtractionMeasurement",
    "Quantisation",
    "GGUFError",
    "ModelProfile",
    "read_profile",
    "Budget",
    "Candidate",
    "Hardware",
    "PlannerConfig",
    "PlanResult",
    "plan_for_hardware",
]
