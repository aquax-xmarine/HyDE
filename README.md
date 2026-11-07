# HyDE: Hypothetical Document Embeddings

Builds HyDE with Haystack and DeepSeek, then tests whether it improves retrieval on the BEIR SciFact benchmark compared with plain query embeddings.

## What is HyDE?

A short query often doesn't look much like the documents it's searching for. HyDE asks an LLM to write several **hypothetical answers** to the query, embeds them, and averages them into one search vector. A fake answer, even if not fully accurate, tends to sit closer to real relevant documents in embedding space than the query does.

HyDE is most useful when the retrieval step isn't good enough, for example when recall is low.

## Pipeline

```
prompt_builder → generator → adapter → embedder → hyde
```

| Component | What it does |
| --- | --- |
| `prompt_builder` | Turns the query into a prompt asking for an answer passage |
| `generator` | DeepSeek writes 5 different hypothetical answers |
| `adapter` | Wraps each answer in a Haystack `Document` |
| `embedder` | Turns each document into a 384-dimensional vector (`all-MiniLM-L6-v2`) |
| `hyde` | Averages the 5 vectors into one search vector |

## Three search methods

| Method | Search vector |
| --- | --- |
| **Plain query** | Embedding of the query itself |
| **HyDE** | Average of the 5 hypothetical passages |
| **HyDE + query** | (5 passages + the original query) / 6, as in the original HyDE paper |

In HyDE + query, the original query acts as an **anchor**, keeping the search tied to what was actually asked in case the generated passages drift.

## Files

| File | Purpose |
| --- | --- |
| `hyde_components.py` | Generator, prompt builder, adapter, embedder, and the custom `RepeatedGenerator` and `HypotheticalDocumentEmbedder` components |
| `hyde_pipeline.py` | Connects the components and runs one example query |
| `hyde_eval.py` | Compares the three methods on SciFact |

## Working around DeepSeek's limit

The original tutorial asked the LLM for 5 answers in one call with `n=5`. DeepSeek only supports `n=1`, so `RepeatedGenerator` wraps the generator, calls it 5 times, and returns all 5 replies, looking like a single 5-answer generator to the rest of the pipeline.

## Results

50 SciFact test queries, `all-MiniLM-L6-v2` embeddings, `deepseek-chat` for generation, and a prompt asking for "a scientific paper passage to support or refute the claim."

| Method | nDCG@10 | Recall@100 |
| --- | --- | --- |
| Plain query | 0.7384 | 0.96 |
| HyDE | 0.7750 | 0.98 |
| HyDE + query | **0.7857** | **1.00** |

## Findings

- **HyDE improved retrieval.** Searching with hypothetical passages raised nDCG@10 by 0.037 over the plain query, about 5%.
- **The anchor helped a little more.** HyDE + query gave the best result, +0.047 nDCG@10, matching the original paper's approach.
- **It improved recall, not just ranking.** With HyDE + query, every correct abstract for these 50 claims appeared in the top 100.
- **The hypothetical answers can be very similar.** For a common question like "What should I do if I have a fever?", four of the five answers began almost identically, even at temperature 0.75. When answers are near-copies, averaging adds little over a single answer.

## Caveats

- **Small sample.** With only 50 queries, part of the difference could be noise. A paired significance test would confirm it.
- **Cost.** HyDE adds 5 LLM calls per query, 250 calls for this run, in exchange for +0.047 nDCG@10.
- **Not comparable with full-test-set results,** since these are only the first 50 of about 300 test queries.

## Setup

```bash
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your-key" > .env
```

`hyde_eval.py` reuses the SciFact data and the `data.py` and `metrics.py` files from the [cross-encoder reranking project](https://github.com/aquax-xmarine/cross-encoder-reranking-scifact), expected at `~/AI-engineering/cross-encoder-reranker-project`. Change `RERANKER_DIR` if yours is elsewhere.

## Run it

```bash
python3 hyde_pipeline.py   # one example query, showing the 5 hypothetical answers
python3 hyde_eval.py       # the SciFact comparison
```

The first evaluation run embeds the corpus once and saves it to `corpus_embeddings.npy`. HyDE vectors are cached in `hyde_vectors.json`, so reruns don't repeat API calls.

## Next steps

- [ ] Run all ~300 SciFact test queries
- [ ] Run a paired significance test on per-query scores
- [ ] Increase diversity with a higher temperature or varied prompt styles
- [ ] Combine HyDE + query retrieval with cross-encoder reranking