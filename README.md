# PAMAE-RAG

PAMAE-RAG is a clean research repository for testing a PAMAE-consistent retrieval core for multi-hop GraphRAG.

Version 1 intentionally implements only the PAMAE logic:

```text
Global search on query-sampled anchor candidates
→ full query-universe objective validation
→ monotone local refinement on the entire query-relevant universe
→ evidence rendering under a token budget
```

The terminal-conditioned evidence-chain posterior is deferred to a later branch after this clean baseline is validated.

## Repository layout

```text
PAMAE-RAG/
├── data/                         # Put datasets here. Kept out of git except tiny smoke data.
│   ├── raw/                      # Original benchmark files.
│   ├── processed/                # Query-local universes and embeddings.
│   ├── indexes/                  # Optional vector/graph indexes.
│   ├── graphs/                   # Optional query graph artifacts.
│   ├── runs/                     # Experiment outputs.
│   └── smoke/                    # Tiny checked-in smoke dataset.
├── configs/                      # YAML configs.
├── docs/                         # Method and experiment notes.
├── scripts/                      # Thin wrappers around package CLI.
├── src/pamae_rag/                # Method implementation.
└── tests/                        # Unit and smoke tests.
```

## Git start

Create an empty remote repository named `PAMAE-RAG`, then:

```bash
git clone git@github.com:<YOUR_GITHUB_ID>/PAMAE-RAG.git
cd PAMAE-RAG
```

Copy this scaffold into the cloned directory. If you start from the zip:

```bash
unzip PAMAE-RAG-clean.zip
cd PAMAE-RAG
git init
git branch -M main
git add .
git commit -m "init PAMAE-RAG clean baseline"
git remote add origin git@github.com:<YOUR_GITHUB_ID>/PAMAE-RAG.git
git push -u origin main
```

## Environment setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate pamae-rag
pip install -e .
pytest -q
```

## Smoke run

```bash
python scripts/validate_data.py --input data/smoke/examples.jsonl
python scripts/run_retrieval.py \
  --config configs/smoke.yaml \
  --input data/smoke/examples.jsonl \
  --output data/runs/smoke/results.jsonl
python scripts/evaluate_retrieval.py \
  --input data/smoke/examples.jsonl \
  --predictions data/runs/smoke/results.jsonl \
  --output data/runs/smoke/metrics.json
```

## Dataset schema

The v1 runner expects one JSON object per query. Put your dataset at, for example:

```text
data/processed/<dataset>/examples.jsonl
```

Each line:

```json
{
  "query_id": "q1",
  "query": "Which city hosts the university where Ada studied?",
  "nodes": [
    {
      "node_id": "q1_c1",
      "text": "Ada studied mathematics at Example College.",
      "node_type": "chunk",
      "relevance": 0.82,
      "embedding": [0.1, 0.2, 0.3],
      "token_count": 12,
      "is_anchor_candidate": true,
      "metadata": {"title": "Ada"}
    }
  ],
  "gold_node_ids": ["q1_c1"]
}
```

`relevance` is a non-negative query relevance score before normalization. Embeddings are required for the default angular metric.

## Core objective

For query-relevant universe \(V_q\), candidate anchors \(\mathcal{A}_q\), relevance mass \(\rho_q\), and metric distance \(d_q\), v1 minimizes:

```latex
\mathcal{L}_q(A)
=
\sum_{v\in V_q}\rho_q(v)\min_{a\in A}d_q(v,a)
+
\lambda_T\sum_{a\in A}T(a)
+
\lambda_k|A|.
```

The same objective is used in sample-level global search, full-universe seed selection, and refinement acceptance.

## Why exact search by default?

The first target is not maximum speed. It is to verify whether PAMAE logic improves support recall without heuristic stage drift. Therefore sample-level k-medoids is exact by default and fails if the combination count exceeds the configured cap. Increase the cap only if latency allows it; otherwise reduce sample size.
