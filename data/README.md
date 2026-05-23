# Data directory

Put datasets here.

Recommended layout:

```text
data/
├── raw/
│   ├── hotpotqa/
│   ├── 2wikimultihopqa/
│   └── musique/
├── processed/
│   ├── hotpotqa/examples.jsonl
│   ├── 2wikimultihopqa/examples.jsonl
│   └── musique/examples.jsonl
├── indexes/
├── graphs/
├── runs/
└── smoke/examples.jsonl
```

The v1 pipeline expects one query-local JSON object per line. See `data/smoke/examples.jsonl`.
