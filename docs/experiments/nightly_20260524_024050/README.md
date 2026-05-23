# Nightly PAMAE Compact Experiment

- Start time: 2026-05-24 02:40:50 KST
- Base commit: 1622cf28705cc72c0f811ba8468fc4004b75ff63
- Branch: experiment/nightly_pamae_compact_20260524_024050
- Conda env: pamae-rag
- CUDA_VISIBLE_DEVICES: 0
- Deadline: 09:00 KST

## Boundary Rules

- Do not change PAMAE objective.
- Do not add stage-specific heuristic scores.
- Do not use gold labels for retrieval/relevance scoring.
- Do not inject gold evidence into retrieval candidates.
- Do not implement terminal-conditioned posterior tonight.
- Do not implement graph-aware metric tonight.
- Commit every meaningful step.
- Do not commit data/raw, data/processed, data/runs.
