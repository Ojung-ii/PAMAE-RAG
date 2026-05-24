from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm


class RawDatasetError(ValueError):
    pass


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_title(title: Any) -> str:
    text = unicodedata.normalize("NFKD", _safe_text(title)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _parse_jsonish_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [value]
    return [str(value)]


def _answer_from_example(example: dict[str, Any]) -> str | None:
    for key in ("answer", "obj", "target", "final_answer"):
        if key in example and example[key] is not None:
            return str(example[key])
    answers = _parse_jsonish_list(example.get("possible_answers"))
    return answers[0] if answers else None


def _possible_answers_from_example(example: dict[str, Any]) -> list[str]:
    out = []
    if example.get("answer") is not None:
        out.append(str(example["answer"]))
    if example.get("obj") is not None:
        out.append(str(example["obj"]))
    out.extend(_parse_jsonish_list(example.get("possible_answers")))
    # Preserve order while deduplicating.
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _context_paragraphs(example: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    context = example.get("context") or []
    if isinstance(context, list):
        for item in context:
            title = ""
            sentences: list[str] = []
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = _safe_text(item[0])
                raw_sentences = item[1]
                if isinstance(raw_sentences, list):
                    sentences = [_safe_text(x) for x in raw_sentences]
                else:
                    sentences = [_safe_text(raw_sentences)]
            elif isinstance(item, dict):
                title = _safe_text(item.get("title"))
                raw_sentences = item.get("sentences", item.get("text", item.get("content", "")))
                if isinstance(raw_sentences, list):
                    sentences = [_safe_text(x) for x in raw_sentences]
                else:
                    sentences = [_safe_text(raw_sentences)]
            if title:
                out[normalize_title(title)] = {
                    "title": title,
                    "text": " ".join(s for s in sentences if s).strip(),
                    "sentences": sentences,
                }
    return out


def _support_titles_and_sentences(example: dict[str, Any]) -> list[dict[str, Any]]:
    supports: list[dict[str, Any]] = []
    for key in ("supporting_facts", "support_facts"):
        raw = example.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, (list, tuple)) and item:
                    supports.append(
                        {
                            "title": _safe_text(item[0]),
                            "sentence_id": item[1] if len(item) > 1 else None,
                            "source": key,
                        }
                    )
                elif isinstance(item, dict):
                    title = item.get("title", item.get("wiki_title", item.get("page")))
                    if title is not None:
                        supports.append(
                            {
                                "title": _safe_text(title),
                                "sentence_id": item.get("sent_id", item.get("sentence_id")),
                                "source": key,
                            }
                        )
    raw_titles = example.get("supporting_facts_titles")
    if isinstance(raw_titles, list):
        for title in raw_titles:
            supports.append({"title": _safe_text(title), "sentence_id": None, "source": "supporting_facts_titles"})
    return [s for s in supports if s["title"]]


def _paragraph_supports(example: dict[str, Any]) -> list[dict[str, Any]]:
    supports: list[dict[str, Any]] = []
    paragraphs = example.get("paragraphs") or []
    if isinstance(paragraphs, list):
        for p in paragraphs:
            if not isinstance(p, dict):
                continue
            if bool(p.get("is_supporting", False)):
                supports.append(
                    {
                        "title": _safe_text(p.get("title")),
                        "text": _safe_text(p.get("text")),
                        "sentence_id": p.get("sent_id", p.get("sentence_id")),
                        "source": "paragraphs",
                    }
                )
    return [s for s in supports if s["title"] or s.get("text")]


def _support_paragraphs(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract supporting paragraphs from common QA formats.

    Supported directly:
    - PopQA-style: example["paragraphs"] = [{title,text,is_supporting}, ...]
    - Hotpot/2Wiki-style: supporting_facts = [[title, sent_id], ...]
      with context = [[title, [sentences...]], ...]

    If no explicit support flag exists, this function returns an empty list. The
    converter still works but retrieval recall cannot be computed reliably.
    """
    supports = _paragraph_supports(example)
    context_by_title = _context_paragraphs(example)
    for item in _support_titles_and_sentences(example):
        norm = normalize_title(item["title"])
        ctx = context_by_title.get(norm, {})
        supports.append(
            {
                "title": item["title"],
                "text": _safe_text(ctx.get("text")),
                "sentence_id": item.get("sentence_id"),
                "source": item.get("source"),
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in supports:
        key = (normalize_title(item.get("title")), _safe_text(item.get("text")), _safe_text(item.get("sentence_id")))
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _support_like_snippets(example: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key, value in example.items():
        low = key.lower()
        if any(token in low for token in ("support", "evidence", "context", "paragraph")):
            out[key] = repr(value)[:1200]
    return out


def _write_schema_debug(dataset_name: str, qa: list[Any], debug_dir: str | Path | None) -> None:
    if debug_dir is None:
        return
    path = Path(debug_dir) / f"debug_schema_{dataset_name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"dataset={dataset_name}", "No support-like fields were extracted from the first processed examples."]
    for i, ex in enumerate(qa[:2]):
        if not isinstance(ex, dict):
            lines.append(f"example {i}: non-object {type(ex).__name__}")
            continue
        lines.append(f"example {i} keys: {sorted(ex.keys())}")
        lines.append(f"example {i} support-like snippets: {_support_like_snippets(ex)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _qid(example: dict[str, Any], index: int) -> str:
    for key in ("query_id", "qid", "id", "_id"):
        if example.get(key) is not None:
            return str(example[key])
    return str(index)


def _question(example: dict[str, Any]) -> str:
    for key in ("query", "question"):
        if example.get(key) is not None:
            return str(example[key])
    raise RawDatasetError(f"Example lacks query/question keys: {list(example.keys())}")


def _fit_dense_embeddings(corpus_texts: list[str], query_texts: list[str], embedding_dim: int, max_features: int) -> tuple[TfidfVectorizer, np.ndarray, sparse.csr_matrix]:
    if not corpus_texts:
        raise RawDatasetError("Corpus is empty")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
    )
    # Fit on corpus plus queries so query-only tokens receive dimensions.
    X_all = vectorizer.fit_transform(corpus_texts + query_texts)
    X_corpus = X_all[: len(corpus_texts)]
    n_features = X_corpus.shape[1]
    if n_features == 0:
        raise RawDatasetError("TF-IDF vocabulary is empty")
    n_components = max(2, min(int(embedding_dim), n_features - 1, len(corpus_texts) - 1))
    if n_components < 2:
        dense = X_corpus.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=13)
        dense = svd.fit_transform(X_corpus).astype(np.float32)
    dense = normalize(dense, norm="l2", axis=1)
    return vectorizer, dense.astype(np.float32), X_corpus


def prepare_raw_qa_corpus_dataset(
    qa_path: str | Path,
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str,
    max_nodes_per_query: int = 600,
    embedding_dim: int = 128,
    max_features: int = 50000,
    limit: int | None = None,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convert raw QA+corpus JSON files into PAMAE-RAG examples.jsonl.

    The output schema matches pamae_rag.data.io.example_from_dict. Each line is a
    query-local universe. Gold support nodes are aligned to the global corpus by
    exact (title, text) match first, then by title fallback.
    """
    qa = _read_json(qa_path)
    corpus = _read_json(corpus_path)
    if not isinstance(qa, list):
        raise RawDatasetError(f"QA file must be a JSON list: {qa_path}")
    if not isinstance(corpus, list):
        raise RawDatasetError(f"Corpus file must be a JSON list: {corpus_path}")
    if limit is not None:
        qa = qa[:limit]

    corpus_titles: list[str] = []
    corpus_texts: list[str] = []
    for i, row in enumerate(corpus):
        if not isinstance(row, dict):
            raise RawDatasetError(f"Corpus item {i} is not an object")
        title = _safe_text(row.get("title"))
        text = _safe_text(row.get("text") or row.get("content"))
        if not title and not text:
            raise RawDatasetError(f"Corpus item {i} has neither title nor text")
        corpus_titles.append(title)
        corpus_texts.append(text)

    query_texts = [_question(x) for x in qa]
    vectorizer, corpus_dense, X_corpus = _fit_dense_embeddings(
        corpus_texts, query_texts, embedding_dim=embedding_dim, max_features=max_features
    )
    X_queries = vectorizer.transform(query_texts)

    exact_index: dict[tuple[str, str], int] = {}
    title_index: dict[str, list[int]] = defaultdict(list)
    for i, (title, text) in enumerate(zip(corpus_titles, corpus_texts, strict=True)):
        exact_index.setdefault((title, text), i)
        title_index[normalize_title(title)].append(i)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_gold = 0
    missing_gold = 0
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for qi, example in enumerate(tqdm(qa, desc=f"prepare {dataset_name}")):
            if not isinstance(example, dict):
                raise RawDatasetError(f"QA item {qi} is not an object")
            qid = _qid(example, qi)
            question = query_texts[qi]
            sims = (X_corpus @ X_queries[qi].T).toarray().ravel()
            if max_nodes_per_query < 1:
                raise RawDatasetError("max_nodes_per_query must be positive")
            top_n = min(max_nodes_per_query, len(corpus_texts))
            if top_n == len(corpus_texts):
                candidate_idxs = np.arange(len(corpus_texts), dtype=np.int64)
            else:
                candidate_idxs = np.argpartition(-sims, kth=top_n - 1)[:top_n]
                candidate_idxs = candidate_idxs[np.argsort(-sims[candidate_idxs])]

            gold_idxs: list[int] = []
            support_metadata: list[dict[str, Any]] = []
            for p in _support_paragraphs(example):
                total_gold += 1
                title = _safe_text(p.get("title"))
                text = _safe_text(p.get("text"))
                sentence_id = p.get("sentence_id")
                support_metadata.append(
                    {
                        "title": title,
                        "sentence_id": sentence_id,
                        "source": p.get("source"),
                    }
                )
                key = (title, text)
                norm_title = normalize_title(title)
                if key in exact_index:
                    gold_idxs.append(exact_index[key])
                elif norm_title in title_index:
                    gold_idxs.append(title_index[norm_title][0])
                else:
                    missing_gold += 1

            merged = list(dict.fromkeys(int(x) for x in candidate_idxs))

            nodes = []
            for ci in merged:
                title = corpus_titles[ci]
                text = corpus_texts[ci]
                node_id = f"{dataset_name}:doc:{ci}"
                score = float(sims[ci])
                nodes.append(
                    {
                        "node_id": node_id,
                        "text": text,
                        "node_type": "chunk",
                        "relevance": max(score, 1e-12),
                        "embedding": corpus_dense[ci].astype(float).tolist(),
                        "token_count": _token_count(text),
                        "is_anchor_candidate": True,
                        "metadata": {"title": title, "corpus_index": ci, "dataset": dataset_name},
                    }
                )

            gold_node_ids = [f"{dataset_name}:doc:{i}" for i in dict.fromkeys(gold_idxs)]
            row = {
                "query_id": qid,
                "query": question,
                "nodes": nodes,
                "gold_node_ids": gold_node_ids,
                "answer": _answer_from_example(example),
                "metadata": {
                    "dataset": dataset_name,
                    "possible_answers": _possible_answers_from_example(example),
                    "raw_id": qid,
                    "num_gold_nodes": len(gold_node_ids),
                    "support_facts": support_metadata,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    if total_gold == 0:
        _write_schema_debug(dataset_name, qa, debug_dir)

    return {
        "dataset": dataset_name,
        "num_examples": written,
        "num_corpus_docs": len(corpus_texts),
        "output_path": str(output_path),
        "embedding_dim_actual": int(corpus_dense.shape[1]),
        "max_nodes_per_query": max_nodes_per_query,
        "total_support_paragraphs_seen": total_gold,
        "missing_support_paragraphs": missing_gold,
    }
