from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class RuntimeTimer:
    started: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, key: str, value_ms: float) -> None:
        self.marks[str(key)] = float(value_ms)

    def elapsed_ms(self) -> float:
        return float((time.perf_counter() - self.started) * 1000.0)


def context_text_hash(texts: Iterable[str]) -> str:
    payload = "\n\n".join(str(text) for text in texts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_runtime_profile(
    *,
    query_id: str,
    variant: str,
    runtime_mode: str,
    stage_diagnostics: dict[str, Any],
    renderer_diagnostics: dict[str, Any],
    total_retrieval_ms: float,
    diagnostics_logging_ms: float = 0.0,
) -> dict[str, Any]:
    def stage_ms(name: str) -> float:
        stage = stage_diagnostics.get(name)
        if not isinstance(stage, dict):
            return 0.0
        value = stage.get("latency_ms", 0.0)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    def render_ms(name: str) -> float:
        value = renderer_diagnostics.get(name, 0.0)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    return {
        "query_id": str(query_id),
        "variant": str(variant),
        "runtime_mode": str(runtime_mode),
        "time_anchor_ms": stage_ms("query_anchor_construction"),
        "time_core_retrieval_ms": stage_ms("candidate_generation") + stage_ms("content_graph_projection"),
        "time_refinement_ms": stage_ms("local_refinement"),
        "time_support_tree_ms": render_ms("time_support_tree_ms"),
        "time_shell1_construction_ms": render_ms("time_shell1_construction_ms"),
        "time_query_embedding_ms": render_ms("time_query_embedding_ms"),
        "time_embedding_lookup_ms": render_ms("time_embedding_lookup_ms"),
        "time_semantic_ordering_ms": render_ms("time_semantic_ordering_ms"),
        "time_rendering_ms": stage_ms("context_rendering"),
        "time_diagnostics_logging_ms": float(diagnostics_logging_ms),
        "time_total_retrieval_ms": float(total_retrieval_ms),
        "support_tree_chunk_count": int(renderer_diagnostics.get("support_tree_chunk_count", 0) or 0),
        "shell1_chunk_count": int(renderer_diagnostics.get("shell1_chunk_count", 0) or 0),
        "candidate_pool_size": int(
            renderer_diagnostics.get("candidate_pool_size", renderer_diagnostics.get("pool_chunk_count", 0)) or 0
        ),
        "rendered_chunk_count": int(renderer_diagnostics.get("final_context_nodes", 0) or 0),
        "rendered_shell1_chunk_count": int(renderer_diagnostics.get("rendered_shell1_chunk_count", 0) or 0),
        "query_embedding_cache_hit_rate": float(renderer_diagnostics.get("query_embedding_cache_hit_rate", 0.0) or 0.0),
        "query_embedding_cache_miss_count": int(renderer_diagnostics.get("query_embedding_cache_miss_count", 0) or 0),
        "candidate_embedding_lookup_count": int(renderer_diagnostics.get("candidate_embedding_lookup_count", 0) or 0),
        "unique_candidate_embedding_count": int(renderer_diagnostics.get("unique_candidate_embedding_count", 0) or 0),
        "duplicate_embedding_lookup_avoided": int(renderer_diagnostics.get("duplicate_embedding_lookup_avoided", 0) or 0),
    }


def aggregate_runtime_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    means = {
        key: float(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows))
        for key in numeric_keys
    }
    time_items = [
        (key, value)
        for key, value in means.items()
        if key.startswith("time_") and key not in {"time_total_retrieval_ms"}
    ]
    top = sorted(time_items, key=lambda item: item[1], reverse=True)[:2]
    return {
        "num_queries": len(rows),
        "mean": means,
        "top_two_contributors": [{"name": key, "mean_ms": value} for key, value in top],
    }


__all__ = ["RuntimeTimer", "aggregate_runtime_profiles", "build_runtime_profile", "context_text_hash"]
