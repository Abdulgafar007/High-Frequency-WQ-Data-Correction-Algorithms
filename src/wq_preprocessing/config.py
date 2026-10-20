from dataclasses import dataclass
from typing import List

WQ_PARAMETERS = ["Turbidity", "EC", "DO", "Temp"]
STORM_IDS = [1, 2, 3, 4]
DEGRADATION_SCENARIOS = [
    "random_missing",
    "block_missing",
    "additive_noise",
    "field_realistic_fault",
]
SEVERITY_LEVELS = ["moderate", "severe"]

@dataclass
class ExperimentConfig:
    wq_parameters: List[str]
    storm_ids: List[int]
    degradation_scenarios: List[str]
    severity_levels: List[str]
    repetitions: int = 30
    random_seed: int = 42

DEFAULT_CONFIG = ExperimentConfig(
    wq_parameters=WQ_PARAMETERS,
    storm_ids=STORM_IDS,
    degradation_scenarios=DEGRADATION_SCENARIOS,
    severity_levels=SEVERITY_LEVELS,
    repetitions=30,
    random_seed=42,
)
