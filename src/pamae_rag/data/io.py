from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample


def _embedding(value: Any, node_id: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Node {node_id!r} has missing or invalid embedding")
    return arr


def node_from_dict(obj: dict[str, Any]) -> EvidenceNode:
    node_id = obj.get("node_id") or obj.get("id") or obj.get("chunk_id")
    if not node_id:
        raise ValueError(f"Node lacks node_id/id/chunk_id: {obj}")
    text = str(obj.get("text") or obj.get("content") or "")
    token_count = int(obj.get("token_count") or obj.get("tokens") or max(1, len(text.split())))
    relevance = obj.get("relevance", obj.get("score", obj.get("query_score", 1.0)))
    return EvidenceNode(
        node_id=str(node_id),
        text=text,
        embedding=_embedding(obj.get("embedding"), str(node_id)),
        relevance=float(relevance if relevance is not None else 1.0),
        token_count=token_count,
        node_type=str(obj.get("node_type", obj.get("type", "chunk"))),
        is_anchor_candidate=bool(obj.get("is_anchor_candidate", True)),
        metadata=dict(obj.get("metadata") or {}),
    )


def example_from_dict(obj: dict[str, Any]) -> QueryExample:
    qid = obj.get("query_id") or obj.get("qid") or obj.get("id")
    if not qid:
        raise ValueError(f"Example lacks query_id/qid/id: {obj}")
    query = obj.get("query") or obj.get("question")
    if query is None:
        raise ValueError(f"Example {qid!r} lacks query/question")
    raw_nodes = obj.get("nodes") or obj.get("chunks") or []
    if not raw_nodes:
        raise ValueError(f"Example {qid!r} has no nodes/chunks")
    nodes = tuple(node_from_dict(x) for x in raw_nodes)
    dim = nodes[0].embedding.shape[0]
    for node in nodes:
        if node.embedding.shape[0] != dim:
            raise ValueError(f"Example {qid!r} has inconsistent embedding dimensions")
    gold = obj.get("gold_node_ids") or obj.get("support_node_ids") or obj.get("support_ids") or []
    metadata = {k: v for k, v in obj.items() if k not in {"query_id", "qid", "id", "query", "question", "nodes", "chunks", "gold_node_ids", "support_node_ids", "support_ids", "answer"}}
    return QueryExample(
        query_id=str(qid),
        query=str(query),
        nodes=nodes,
        gold_node_ids=frozenset(str(x) for x in gold),
        answer=obj.get("answer"),
        metadata=metadata,
    )


def read_jsonl(path: str | Path, limit: int | None = None) -> list[QueryExample]:
    path = Path(path)
    out: list[QueryExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(example_from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            if limit is not None and len(out) >= limit:
                break
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
