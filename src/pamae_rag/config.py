from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

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
class DistanceWeightsConfig:
    semantic: float = 0.7
    graph: float = 0.3


@dataclass(frozen=True)
class GraphEdgeLengthsConfig:
    same_canonical_title: float = 0.25
    title_mention: float = 0.50
    shared_query_span: float = 0.75
    shared_entity: float = 1.0
    entity_fact_bridge: float = 1.0


@dataclass(frozen=True)
class GraphBackboneConfig:
    enabled: bool = False
    mode: str = "none"
    k: int = 4
    length_mode: str = "semantic_distance"
    max_edges_per_node: int = 32


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool = False
    source: str = "legacy_query"
    disconnected_distance: float = 2.0
    max_edges_per_node: int = 32
    edge_lengths: GraphEdgeLengthsConfig = field(default_factory=GraphEdgeLengthsConfig)
    backbone: GraphBackboneConfig = field(default_factory=GraphBackboneConfig)


@dataclass(frozen=True)
class PamaeConfig:
    retrieval_variant: str = "sample_full_validation_refine"
    renderer: str = "old"
    relevance_mode: str = "current"
    relevance_weights: dict[str, float] = field(default_factory=dict)
    distance_mode: str = "semantic"
    distance_weights: DistanceWeightsConfig = field(default_factory=DistanceWeightsConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
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
    max_context_nodes: int | None = None
    strict_context_budget: bool = False
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
    if cls is PamaeConfig:
        values = dict(values)
        if "distance_weights" in values and isinstance(values["distance_weights"], dict):
            values["distance_weights"] = _make(DistanceWeightsConfig, values["distance_weights"])
        if "graph" in values and isinstance(values["graph"], dict):
            graph_values = dict(values["graph"])
            if "edge_lengths" in graph_values and isinstance(graph_values["edge_lengths"], dict):
                graph_values["edge_lengths"] = _make(GraphEdgeLengthsConfig, graph_values["edge_lengths"])
            if "backbone" in graph_values and isinstance(graph_values["backbone"], dict):
                graph_values["backbone"] = _make(GraphBackboneConfig, graph_values["backbone"])
            values["graph"] = _make(GraphConfig, graph_values)
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
    relevance_modes = {
        "current",
        "title_aware",
        "entity_title_aware",
        "hybrid_title_semantic",
        "diagnostic_subject_title",
    }
    if cfg.pamae.retrieval_variant not in retrieval_variants:
        raise ValueError(f"pamae.retrieval_variant must be one of {sorted(retrieval_variants)}")
    if cfg.pamae.renderer not in renderers:
        raise ValueError(f"pamae.renderer must be one of {sorted(renderers)}")
    if cfg.pamae.relevance_mode not in relevance_modes:
        raise ValueError(f"pamae.relevance_mode must be one of {sorted(relevance_modes)}")
    distance_modes = {"semantic", "graph_sp", "hybrid_sem_graph"}
    if cfg.pamae.distance_mode not in distance_modes:
        raise ValueError(f"pamae.distance_mode must be one of {sorted(distance_modes)}")
    if cfg.pamae.distance_weights.semantic < 0 or cfg.pamae.distance_weights.graph < 0:
        raise ValueError("pamae.distance_weights values must be nonnegative")
    if cfg.pamae.graph.disconnected_distance < 0:
        raise ValueError("pamae.graph.disconnected_distance must be nonnegative")
    if cfg.pamae.graph.source not in {"legacy_query", "content"}:
        raise ValueError("pamae.graph.source must be one of ['content', 'legacy_query']")
    if cfg.pamae.graph.max_edges_per_node < 0:
        raise ValueError("pamae.graph.max_edges_per_node must be nonnegative")
    for key, value in cfg.pamae.graph.edge_lengths.__dict__.items():
        if float(value) < 0:
            raise ValueError(f"pamae.graph.edge_lengths.{key} must be nonnegative")
    if cfg.pamae.graph.backbone.mode not in {"none", "knn", "mutual_knn"}:
        raise ValueError("pamae.graph.backbone.mode must be one of ['knn', 'mutual_knn', 'none']")
    if cfg.pamae.graph.backbone.k < 1:
        raise ValueError("pamae.graph.backbone.k must be positive")
    if cfg.pamae.graph.backbone.max_edges_per_node < 0:
        raise ValueError("pamae.graph.backbone.max_edges_per_node must be nonnegative")
    if cfg.pamae.graph.backbone.length_mode != "semantic_distance":
        raise ValueError("pamae.graph.backbone.length_mode must be 'semantic_distance'")
    allowed_relevance_weights = {"lexical", "title", "entity_title", "semantic"}
    unknown_weights = sorted(set(cfg.pamae.relevance_weights) - allowed_relevance_weights)
    if unknown_weights:
        raise ValueError(f"pamae.relevance_weights has unknown keys: {unknown_weights}")
    for key, value in cfg.pamae.relevance_weights.items():
        if float(value) < 0:
            raise ValueError(f"pamae.relevance_weights.{key} must be nonnegative")
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
    if cfg.pamae.max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")
    if cfg.pamae.max_context_nodes is not None and cfg.pamae.max_context_nodes < 0:
        raise ValueError("max_context_nodes must be nonnegative or null")
    if cfg.pamae.max_context_nodes and cfg.pamae.max_context_nodes < cfg.pamae.k:
        msg = "pamae.max_context_nodes is smaller than pamae.k; anchors may exceed the node budget"
        if cfg.pamae.strict_context_budget:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    if cfg.universe.max_nodes < 1:
        raise ValueError("universe.max_nodes must be positive")
    if not 0 < cfg.universe.min_relevance_mass <= 1:
        raise ValueError("universe.min_relevance_mass must be in (0, 1]")
    if cfg.distance.metric not in {"angular", "cosine"}:
        raise ValueError("distance.metric must be 'angular' or 'cosine'")
