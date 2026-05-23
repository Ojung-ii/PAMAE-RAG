from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "pamae_rag_v1"
    seed: int = 42
    limit_queries: int | None = None


@dataclass(frozen=True)
class UniverseConfig:
    max_nodes: int = 600
    min_relevance_mass: float = 0.98
    anchor_node_types: tuple[str, ...] = ("chunk", "sentence")


@dataclass(frozen=True)
class DistanceConfig:
    metric: str = "angular"


@dataclass(frozen=True)
class PamaeConfig:
    retrieval_variant: str = "sample_full_validation_refine"
    renderer: str = "old"
    relevance_mode: str = "current"
    k: int = 3
    k_max: int = 4
    auto_k: bool = False
    lambda_k: float = 0.0
    num_samples: int = 5
    sample_size_per_k: int = 12
    sample_size_cap: int = 48
    sample_uniform_mix: float = 0.15
    require_exact_sample_search: bool = True
    max_exact_combinations: int = 1_000_000
    refinement_iters: int = 1
    renderer_gamma: float = 0.0
    token_weight: float = 0.03
    anchor_penalty: float = 0.0
    max_context_tokens: int = 1200
    evidence_per_anchor: int = 2


@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    pamae: PamaeConfig = field(default_factory=PamaeConfig)

    @property
    def seed(self) -> int:
        return self.experiment.seed


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section {key!r} must be a mapping")
    return value


def _make(cls, values: dict[str, Any]):
    allowed = set(cls.__dataclass_fields__.keys())
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    if cls is UniverseConfig and "anchor_node_types" in values:
        values = dict(values)
        values["anchor_node_types"] = tuple(values["anchor_node_types"])
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg = AppConfig(
        experiment=_make(ExperimentConfig, _section(raw, "experiment")),
        universe=_make(UniverseConfig, _section(raw, "universe")),
        distance=_make(DistanceConfig, _section(raw, "distance")),
        pamae=_make(PamaeConfig, _section(raw, "pamae")),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    retrieval_variants = {
        "top_rho",
        "sample_only",
        "sample_full_validation",
        "sample_full_validation_refine",
        "sample_full_validation_refine_cell_renderer",
        "adaptive_k",
    }
    renderers = {"old", "anchor_only", "nearest", "cell_top_rho", "global_top_rho"}
    relevance_modes = {"current", "title_aware", "diagnostic_subject_title"}
    if cfg.pamae.retrieval_variant not in retrieval_variants:
        raise ValueError(f"pamae.retrieval_variant must be one of {sorted(retrieval_variants)}")
    if cfg.pamae.renderer not in renderers:
        raise ValueError(f"pamae.renderer must be one of {sorted(renderers)}")
    if cfg.pamae.relevance_mode not in relevance_modes:
        raise ValueError(f"pamae.relevance_mode must be one of {sorted(relevance_modes)}")
    if cfg.pamae.k < 1 or cfg.pamae.k_max < 1:
        raise ValueError("k and k_max must be positive")
    if cfg.pamae.lambda_k < 0:
        raise ValueError("lambda_k must be nonnegative")
    if cfg.pamae.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if cfg.pamae.sample_size_per_k < 1 or cfg.pamae.sample_size_cap < 1:
        raise ValueError("sample size controls must be positive")
    if not 0 <= cfg.pamae.sample_uniform_mix <= 1:
        raise ValueError("sample_uniform_mix must be in [0, 1]")
    if cfg.pamae.max_exact_combinations < 1:
        raise ValueError("max_exact_combinations must be positive")
    if cfg.pamae.renderer_gamma < 0:
        raise ValueError("renderer_gamma must be nonnegative")
    if cfg.pamae.token_weight < 0 or cfg.pamae.anchor_penalty < 0:
        raise ValueError("objective penalties must be nonnegative")
    if cfg.universe.max_nodes < 1:
        raise ValueError("universe.max_nodes must be positive")
    if not 0 < cfg.universe.min_relevance_mass <= 1:
        raise ValueError("universe.min_relevance_mass must be in (0, 1]")
    if cfg.distance.metric not in {"angular", "cosine"}:
        raise ValueError("distance.metric must be 'angular' or 'cosine'")
