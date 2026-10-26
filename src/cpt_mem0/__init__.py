"""cpt_mem0 - self-contained local mem0 service for consumer GPUs.

The package is importable without a GPU, without network access and without
any model being present on disk. Everything that touches hardware lives
behind an explicit call; the planner is pure arithmetic over measured
constants.
"""

__version__ = "0.3.0"

import os as _os

# mem0's telemetry is off, and it has to be off **before mem0 is
# imported** - the flag is read once, when the module loads.
#
# Two reasons, and the second one is not about privacy:
#
# 1. The app promises no outbound requests. With the telemetry on,
#    mem0 sends events to PostHog.
#
# 2. **The telemetry opens a second, global Qdrant client** under
#    ``~/.mem0/migrations_qdrant``. Qdrant's embedded mode allows one
#    client per directory, so that path is an exclusive lock. With
#    the telemetry on, Cpt Mem0 cannot start while **any other mem0
#    process** on the machine holds it - including the Hermes stack.
#    The error names a migrations folder and says nothing about the
#    real cause.
#
# The user's own mem0 stack keeps its telemetry as it is. This only
# changes what this app does - and it is forced, not defaulted. A
# global MEM0_TELEMETRY=True in the environment would otherwise put
# the app back into the exclusive lock and send the events out.
_os.environ["MEM0_TELEMETRY"] = "False"

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
