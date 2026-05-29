from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pamae_rag.data.schema import QueryExample
from pamae_rag.semantic.embedding_store import EmbeddingStore

NV_EMBED_V2 = "nvidia/NV-Embed-v2"
QWEN3_EMBED_8B = "Qwen/Qwen3-Embedding-8B"


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    model_slug: str
    local_found: bool
    revision: str | None
    snapshot_path: str | None
    config_found: bool
    tokenizer_found: bool
    weights_found: bool
    usable_for_generation: bool
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_slug": self.model_slug,
            "local_found": self.local_found,
            "revision": self.revision,
            "snapshot_path": self.snapshot_path,
            "config_found": self.config_found,
            "tokenizer_found": self.tokenizer_found,
            "weights_found": self.weights_found,
            "usable_for_generation": self.usable_for_generation,
            "reason": self.reason,
        }


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace("-", "_")


def _candidate_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for env_key in ("HF_HOME", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value))
            roots.append(Path(value) / "hub")
    roots.extend(
        [
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("/home/dilab/.cache/huggingface"),
            Path("/home/dilab/.cache/huggingface/hub"),
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _repo_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _read_revision(repo_dir: Path) -> str | None:
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        value = ref.read_text(encoding="utf-8").strip()
        if value:
            return value
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        dirs = sorted(path.name for path in snapshots.iterdir() if path.is_dir())
        if dirs:
            return dirs[-1]
    return None


def _find_snapshot(model_id: str) -> tuple[Path | None, str | None]:
    repo_name = _repo_dir_name(model_id)
    for root in _candidate_cache_roots():
        repo_dir = root / repo_name
        if not repo_dir.exists():
            continue
        revision = _read_revision(repo_dir)
        if revision:
            snapshot = repo_dir / "snapshots" / revision
            if snapshot.exists():
                return snapshot, revision
        snapshots = repo_dir / "snapshots"
        if snapshots.exists():
            for snapshot in sorted((path for path in snapshots.iterdir() if path.is_dir()), reverse=True):
                return snapshot, snapshot.name
    return None, None


def inspect_local_model(model_id: str) -> ModelCandidate:
    snapshot, revision = _find_snapshot(model_id)
    if snapshot is None:
        return ModelCandidate(
            model_id=model_id,
            model_slug=_model_slug(model_id),
            local_found=False,
            revision=None,
            snapshot_path=None,
            config_found=False,
            tokenizer_found=False,
            weights_found=False,
            usable_for_generation=False,
            reason="no local Hugging Face snapshot found",
        )
    config_found = (snapshot / "config.json").exists()
    tokenizer_found = any((snapshot / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    weights_found = bool(list(snapshot.glob("*.safetensors")) or list(snapshot.glob("pytorch_model*.bin")))
    usable = bool(config_found and tokenizer_found and weights_found)
    missing = [
        name
        for name, found in (
            ("config", config_found),
            ("tokenizer", tokenizer_found),
            ("weights", weights_found),
        )
        if not found
    ]
    return ModelCandidate(
        model_id=model_id,
        model_slug=_model_slug(model_id),
        local_found=True,
        revision=revision,
        snapshot_path=str(snapshot),
        config_found=config_found,
        tokenizer_found=tokenizer_found,
        weights_found=weights_found,
        usable_for_generation=usable,
        reason="local snapshot is complete" if usable else "missing " + ", ".join(missing),
    )


def audit_existing_example_embeddings(examples: Sequence[QueryExample]) -> dict[str, Any]:
    dims: set[int] = set()
    chunk_found = False
    query_found = False
    chunk_count = 0
    for example in examples:
        store = EmbeddingStore.from_example(example)
        diagnostics = store.diagnostics()
        if diagnostics.chunk_count:
            chunk_count += diagnostics.chunk_count
            chunk_found = chunk_found or diagnostics.chunk_embedding_coverage > 0.0
            if diagnostics.embedding_dim:
                dims.add(diagnostics.embedding_dim)
        query_found = query_found or diagnostics.query_embedding_available
    dim = min(dims) if dims else 0
    reason = "missing query embeddings or unknown provenance"
    return {
        "existing_chunk_embedding_found": chunk_found,
        "existing_chunk_embedding_dim": dim,
        "existing_chunk_embedding_dim_values": sorted(dims),
        "existing_chunk_count": chunk_count,
        "existing_model_id": None,
        "existing_model_revision": None,
        "existing_pooling": None,
        "existing_normalization": None,
        "existing_query_embedding_found": query_found,
        "existing_cache_usable_for_semantic_query_chunk": False,
        "reason": reason,
    }


def select_embedding_model() -> dict[str, Any]:
    nv = inspect_local_model(NV_EMBED_V2)
    qwen = inspect_local_model(QWEN3_EMBED_8B)
    if nv.usable_for_generation:
        selected = nv
        selection_reason = "valid local NV-Embed-v2 snapshot supports query and chunk generation"
    elif qwen.usable_for_generation:
        selected = qwen
        selection_reason = "NV-Embed-v2 unavailable or invalid; local Qwen3-Embedding-8B snapshot is usable"
    else:
        selected = None
        selection_reason = "no usable local NV-Embed-v2 or Qwen3-Embedding-8B snapshot"
    return {
        "nv_embed_v2": nv.to_json(),
        "qwen3_embedding_8b": qwen.to_json(),
        "selected_model": selected.to_json() if selected is not None else None,
        "selected_model_id": selected.model_id if selected is not None else None,
        "selected_model_revision": selected.revision if selected is not None else None,
        "selected_model_path": selected.snapshot_path if selected is not None else None,
        "selection_reason": selection_reason,
    }


def write_audit_markdown(audit: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = audit.get("model_selection", {}).get("selected_model") or {}
    existing = audit.get("existing_embeddings", {})
    lines = [
        "# Embedding Space Audit",
        "",
        "## Existing Processed Embeddings",
        "",
        f"- Chunk embedding found: `{existing.get('existing_chunk_embedding_found', False)}`",
        f"- Chunk embedding dim: `{existing.get('existing_chunk_embedding_dim', 0)}`",
        f"- Query embedding found: `{existing.get('existing_query_embedding_found', False)}`",
        f"- Existing cache usable: `{existing.get('existing_cache_usable_for_semantic_query_chunk', False)}`",
        f"- Reason: {existing.get('reason', '')}",
        "",
        "## Local Model Selection",
        "",
        f"- Selected model: `{selected.get('model_id')}`" if selected else "- Selected model: `None`",
        f"- Revision: `{selected.get('revision')}`" if selected else "- Revision: `None`",
        f"- Snapshot path: `{selected.get('snapshot_path')}`" if selected else "- Snapshot path: `None`",
        f"- Reason: {audit.get('model_selection', {}).get('selection_reason', '')}",
        "",
        "## Raw JSON",
        "",
        "```json",
        json.dumps(audit, indent=2, sort_keys=True),
        "```",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "NV_EMBED_V2",
    "QWEN3_EMBED_8B",
    "ModelCandidate",
    "audit_existing_example_embeddings",
    "inspect_local_model",
    "select_embedding_model",
    "write_audit_markdown",
]
