"""Compare plain query embeddings with HyDE on BEIR SciFact.

Reuses the HyDE components from hyde_components.py, and the data loaders
and metrics from the cross-encoder reranker project.
"""

import json
import os
import sys

import numpy as np
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from sentence_transformers import SentenceTransformer

from hyde_components import (
    generator,
    RepeatedGenerator,
    adapter,
    embedder,
    HypotheticalDocumentEmbedder,
)

# Reuse data.py and metrics.py from the reranker project
RERANKER_DIR = os.path.expanduser("~/AI-engineering/cross-encoder-reranker-project")
sys.path.append(RERANKER_DIR)
from data import load_corpus, load_queries, load_qrels  # noqa: E402
from metrics import ndcg_at_k, recall_at_k  # noqa: E402

SCIFACT = os.path.join(RERANKER_DIR, "scifact")
NUM_QUERIES = 50          # each query costs 5 DeepSeek calls
CACHE_FILE = "hyde_vectors.json"
CORPUS_EMB_FILE = "corpus_embeddings.npy"

# --- Load data ---
corpus = load_corpus(f"{SCIFACT}/corpus.jsonl")
queries = load_queries(f"{SCIFACT}/queries.jsonl")
qrels = load_qrels(f"{SCIFACT}/qrels/test.tsv")
doc_ids = list(corpus.keys())
test_qids = list(qrels.keys())[:NUM_QUERIES]

# --- 1. Embed the corpus once, with the same model HyDE uses ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(CORPUS_EMB_FILE):
    corpus_emb = np.load(CORPUS_EMB_FILE)
else:
    print("Embedding the corpus (one-time, a few minutes)...")
    corpus_emb = model.encode(
        [corpus[d] for d in doc_ids],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(CORPUS_EMB_FILE, corpus_emb)


def search(vector, k=100):
    """Return the top-k document IDs by cosine similarity."""
    v = np.array(vector, dtype=np.float32)
    v = v / np.linalg.norm(v)
    scores = corpus_emb @ v
    top = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in top]


# --- 2. HyDE pipeline, with a prompt suited to scientific claims ---
prompt_builder = ChatPromptBuilder(
    template=[
        ChatMessage.from_user(
            "Please write a scientific paper passage to support or refute the claim.\n"
            "Claim: {{question}}\n"
            "Passage:"
        ),
    ],
    required_variables="*",
)

pipeline = Pipeline()
pipeline.add_component(name="prompt_builder", instance=prompt_builder)
pipeline.add_component(name="generator", instance=RepeatedGenerator(generator, n=5))
pipeline.add_component(name="adapter", instance=adapter)
pipeline.add_component(name="embedder", instance=embedder)
pipeline.add_component(name="hyde", instance=HypotheticalDocumentEmbedder())

pipeline.connect("prompt_builder.prompt", "generator.messages")
pipeline.connect("generator.replies", "adapter.answers")
pipeline.connect("adapter.output", "embedder.documents")
pipeline.connect("embedder.documents", "hyde.documents")

# Cache HyDE vectors, so a rerun or interruption doesn't repeat API calls
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)

for n, qid in enumerate(test_qids, 1):
    if qid in cache:
        continue
    result = pipeline.run(data={"prompt_builder": {"question": queries[qid]}})
    cache[qid] = result["hyde"]["hypothetical_embedding"]
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"HyDE vectors: {n}/{len(test_qids)}")

# --- 3. Compare the three search methods ---
methods = {"Plain query": [], "HyDE": [], "HyDE + query": []}
scores = {name: {"ndcg": [], "recall": []} for name in methods}

for qid in test_qids:
    relevant = qrels[qid]
    query_vec = model.encode(queries[qid], normalize_embeddings=True)
    hyde_vec = np.array(cache[qid])

    vectors = {
        "Plain query": query_vec,
        "HyDE": hyde_vec,
        # As in the HyDE paper: average the query with the 5 hypothetical documents
        "HyDE + query": (5 * hyde_vec + query_vec) / 6,
    }

    for name, vec in vectors.items():
        ranked = search(vec)
        scores[name]["ndcg"].append(ndcg_at_k(ranked, relevant, 10))
        scores[name]["recall"].append(recall_at_k(ranked, relevant, 100))

print(f"\nResults on {len(test_qids)} SciFact test queries")
print(f"{'Method':<14} {'nDCG@10':>8} {'Recall@100':>11}")
for name in methods:
    print(f"{name:<14} {np.mean(scores[name]['ndcg']):>8.4f} "
          f"{np.mean(scores[name]['recall']):>11.4f}")