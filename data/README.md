# Data layout

Raw files are kept out of git. Put each dataset under `data/raw/<dataset>/`.

Recommended raw layout:

```text
data/raw/popqa/popqa.json
data/raw/popqa/popqa_corpus.json

data/raw/hotpotqa/hotpotqa.json
data/raw/hotpotqa/hotpotqa_corpus.json

data/raw/2wikimultihopqa/2wikimultihopqa.json
data/raw/2wikimultihopqa/2wikimultihopqa_corpus.json

data/raw/musique/musique.json
data/raw/musique/musique_corpus.json
```

Processed query-local universes go here:

```text
data/processed/<dataset>/examples.jsonl
```

The raw QA+corpus adapter supports JSON-list QA files with `question` and optional `paragraphs[{title,text,is_supporting}]`, plus JSON-list corpus files with `title` and `text`.
