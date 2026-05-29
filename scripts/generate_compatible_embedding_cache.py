#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids
from pamae_rag.graph.distances import build_distance_matrix
from pamae_rag.graph.graph_distance import build_graph_aware_distance_matrix
from pamae_rag.graph.universe import select_universe_by_mass
from pamae_rag.objective.relevance_mass import normalize_relevance_scores, relevance_scores
from pamae_rag.pipeline import run_query_pamae
from pamae_rag.selection.basin_preserving import query_anchor_indices
from pamae_rag.semantic.compatible_embedding_cache import (
    CompatibleEmbeddingCache,
    default_cache_root,
    make_metadata,
)
from pamae_rag.semantic.embedding_provenance import select_embedding_model
from pamae_rag.semantic.semantic_candidate_ordering import build_semantic_graph_pool, node_id


class NvEmbedV2Embedder:
    model_id = "nvidia/NV-Embed-v2"
    normalized = True
    pooling = "nv_embed_encode"

    def __init__(
        self,
        *,
        model_path: str,
        model_revision: str,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model_revision = model_revision
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        )
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()
        self.embedding_dim = int(getattr(getattr(self.model, "config", None), "hidden_size", 4096))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch.nn.functional as F

        outputs: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tensor = self.model.encode(batch, instruction="", max_length=self.max_length)
                tensor = F.normalize(tensor.float(), p=2, dim=1)
                outputs.append(tensor.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, self.embedding_dim), dtype=np.float32)


def _indices_from_node_ids(nodes, node_ids: Iterable[str]) -> list[int]:
    by_id = {str(node.node_id): idx for idx, node in enumerate(nodes)}
    return [by_id[str(node_id)] for node_id in node_ids if str(node_id) in by_id]


def _required_ids_for_example(example, cfg, seed_offset: int) -> tuple[set[str], dict[str, Any]]:
    retrieval = run_query_pamae(example, cfg, seed_offset=seed_offset)
    row = retrieval.to_json()
    nodes = select_universe_by_mass(
        example.nodes,
        max_nodes=cfg.universe.max_nodes,
        min_relevance_mass=cfg.universe.min_relevance_mass,
    )
    embeddings = np.vstack([node.embedding for node in nodes])
    semantic_distance_matrix = build_distance_matrix(embeddings, metric=cfg.distance.metric)
    graph_result = build_graph_aware_distance_matrix(
        nodes,
        example.query,
        semantic_distance_matrix,
        distance_mode=cfg.pamae.distance_mode,
        distance_weights={
            "semantic": cfg.pamae.distance_weights.semantic,
            "graph": cfg.pamae.distance_weights.graph,
        },
        graph_config=cfg.pamae.graph,
    )
    graph_distance_matrix = graph_result.graph_distance_matrix
    if graph_distance_matrix is None:
        graph_distance_matrix = graph_result.distance_matrix
    graph_diagnostics = dict(graph_result.diagnostics)
    disconnected_distance = float(graph_diagnostics.get("graph_disconnected_distance", 2.0))
    candidates = [
        idx
        for idx, node in enumerate(nodes)
        if node.is_anchor_candidate and (not cfg.universe.anchor_node_types or node.node_type in cfg.universe.anchor_node_types)
    ]
    if not candidates:
        candidates = list(range(len(nodes)))
    rho_scores = relevance_scores(
        nodes,
        mode=cfg.pamae.relevance_mode,
        query=example.query,
        query_metadata=example.metadata,
        weights=cfg.pamae.relevance_weights,
    )
    rho = normalize_relevance_scores(rho_scores)
    diagnostics = row.get("diagnostics", {})
    selected = _indices_from_node_ids(nodes, row.get("anchor_node_ids", []))
    query_anchors = _indices_from_node_ids(nodes, diagnostics.get("diagnostic_query_anchor_node_ids", []))
    if not query_anchors:
        query_anchors = query_anchor_indices(candidates, rho, max(1, len(selected) or 1))
    pool = build_semantic_graph_pool(
        nodes=nodes,
        selected_medoids=selected,
        query_anchors=query_anchors,
        distance_matrix=graph_distance_matrix,
        disconnected_distance=disconnected_distance,
    )
    required = {
        node_id(nodes, idx)
        for idx in sorted(pool.support_tree_chunks | pool.shell1_chunks | pool.shell2_chunks)
    }
    required.update(str(node_id) for node_id in row.get("context_node_ids", []))
    required.update(set(answer_containing_chunk_ids(example, nodes)) & set(diagnostics.get("projected_node_ids", [])))
    return required, {
        "query_id": example.query_id,
        "support_tree_chunk_count": len(pool.support_tree_chunks),
        "shell1_chunk_count": len(pool.shell1_chunks),
        "shell2_chunk_count": len(pool.shell2_chunks),
        "required_chunk_count": len(required),
    }


def generate_cache(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    dataset: str,
    limit: int,
    batch_size: int,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=limit)
    selection = select_embedding_model()["selected_model"]
    if not selection or selection.get("model_id") != "nvidia/NV-Embed-v2":
        raise RuntimeError("No usable local NV-Embed-v2 model was selected by the provenance audit")
    metadata = make_metadata(
        model_id=str(selection["model_id"]),
        model_revision=str(selection["revision"]),
        embedding_dim=4096,
        dataset=dataset,
        pooling="nv_embed_encode",
        model_path=str(selection["snapshot_path"]),
    )
    cache_root = default_cache_root(model_id=metadata.model_id, dataset=dataset)
    cache = CompatibleEmbeddingCache.create(cache_root, metadata)
    started = time.perf_counter()
    embedder = NvEmbedV2Embedder(
        model_path=str(selection["snapshot_path"]),
        model_revision=str(selection["revision"]),
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    by_query: list[dict[str, Any]] = []
    all_required: set[str] = set()
    nodes_by_id: dict[str, Any] = {}
    for idx, example in enumerate(examples):
        required, diag = _required_ids_for_example(example, cfg, idx)
        by_query.append(diag)
        all_required.update(required)
        for node in example.nodes:
            if str(node.node_id) in required:
                nodes_by_id[str(node.node_id)] = node
        cache.ensure_query(example, embedder)
    cache.ensure_chunks(list(nodes_by_id.values()), sorted(all_required), embedder)
    coverage = cache.coverage(query_ids=[example.query_id for example in examples], chunk_ids=sorted(all_required))
    summary = {
        "dataset": dataset,
        "cache_root": str(cache_root),
        "model_id": metadata.model_id,
        "model_revision": metadata.model_revision,
        "model_path": metadata.model_path,
        "embedding_dim": metadata.embedding_dim,
        "normalized": metadata.normalized,
        "pooling": metadata.pooling,
        "chunk_text_format": metadata.chunk_text_format,
        "query_text_format": metadata.query_text_format,
        "uses_official_model_instruction_wrapper": metadata.uses_official_model_instruction_wrapper,
        "instruction_template": metadata.instruction_template,
        "num_queries": len(examples),
        "required_chunk_count": len(all_required),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": embedder.device,
        "batch_size": batch_size,
        "query_diagnostics": by_query,
        **coverage,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compatible_embedding_cache_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compatible query/chunk embeddings for semantic diagnostics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    summary = generate_cache(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        dataset=args.dataset,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
