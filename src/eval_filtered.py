#!/usr/bin/env python3
"""
Evaluation with query-specific corpus filtering - Fixed version.
Each query is only searched against its relevant documents from qrels.
"""

import json
import logging
import os
import pathlib
from pathlib import Path

import torch
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

DATA_DIR = Path(__file__).parent.parent / "data" / "beir_data"

corpus_path = DATA_DIR / "corpus.jsonl"
query_path = DATA_DIR / "query.jsonl"
qrels_path = DATA_DIR / "qrels.tsv"

print("Loading data...")
corpus, queries, qrels = GenericDataLoader(
    corpus_file=str(corpus_path), query_file=str(query_path), qrels_file=str(qrels_path)
).load_custom()

print(f"Loaded {len(queries)} queries and {len(corpus)} corpus documents")
print(f"Loaded {sum(len(docs) for docs in qrels.values())} qrels entries")

# Initialize the model
model = DRES(
    models.SentenceBERT(
        "Qwen/Qwen3-Embedding-0.6B",
        trust_remote_code=True,
    ),
    batch_size=16,
)

print("\n=== QUERY-SPECIFIC CORPUS RETRIEVAL ===")
print(
    "Each query will ONLY search within its relevant+irrelevant documents from qrels\n"
)


def custom_search_single_query(model, query_text, query_specific_corpus, top_k=10):
    """
    Custom search function that handles small corpus sizes properly.
    """
    if not query_specific_corpus:
        return {}

    # Encode query
    query_embedding = model.model.encode_queries(
        [query_text], batch_size=1, show_progress_bar=False, convert_to_tensor=True
    )

    # Prepare corpus
    corpus_ids = list(query_specific_corpus.keys())
    corpus_texts = [query_specific_corpus[cid] for cid in corpus_ids]

    # Encode corpus
    corpus_embeddings = model.model.encode_corpus(
        corpus_texts,
        batch_size=min(16, len(corpus_texts)),
        show_progress_bar=False,
        convert_to_tensor=True,
    )

    # Compute similarity scores
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.unsqueeze(0)
    if len(corpus_embeddings.shape) == 1:
        corpus_embeddings = corpus_embeddings.unsqueeze(0)

    # Cosine similarity
    query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    corpus_norm = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
    cos_scores = torch.mm(query_norm, corpus_norm.transpose(0, 1))

    # Get top-k results
    k = min(top_k, len(corpus_ids))
    if k == 0:
        return {}

    # Handle case where cos_scores might be 1D
    if len(cos_scores.shape) == 1:
        cos_scores = cos_scores.unsqueeze(0)

    top_k_values, top_k_idx = torch.topk(cos_scores[0], k, largest=True, sorted=True)

    # Build results dictionary
    results = {}
    for idx, score in zip(top_k_idx.cpu().tolist(), top_k_values.cpu().tolist()):
        results[corpus_ids[idx]] = float(score)

    return results


# Build query-specific corpus for each query
query_specific_results = {}
query_stats = []
failed_queries = []

for i, (query_id, query_text) in enumerate(queries.items(), 1):
    if i % 100 == 0:
        print(f"Processing query {i}/{len(queries)}...")

    # Get ALL documents in qrels for this query (both relevant and irrelevant)
    query_docs_in_qrels = qrels.get(query_id, {})

    if not query_docs_in_qrels:
        print(f"Warning: No qrels found for query {query_id}, skipping...")
        continue

    # Filter corpus to only documents in qrels for this query
    query_specific_corpus = {
        doc_id: corpus[doc_id]
        for doc_id in query_docs_in_qrels.keys()
        if doc_id in corpus
    }

    # Count relevant vs irrelevant
    relevant_count = sum(1 for score in query_docs_in_qrels.values() if score > 0)
    irrelevant_count = len(query_docs_in_qrels) - relevant_count

    query_stats.append(
        {
            "query_id": query_id,
            "total_docs_in_qrels": len(query_docs_in_qrels),
            "relevant_docs": relevant_count,
            "irrelevant_docs": irrelevant_count,
            "corpus_size": len(query_specific_corpus),
        }
    )

    # Skip if no corpus documents found
    if not query_specific_corpus:
        print(f"Warning: No corpus documents found for query {query_id}")
        continue

    try:
        # Use custom search function
        results = custom_search_single_query(
            model, query_text, query_specific_corpus, top_k=1000
        )
        query_specific_results[query_id] = results
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        print(f"  Corpus size: {len(query_specific_corpus)}")
        failed_queries.append(query_id)
        continue

print(f"\nProcessed {len(query_specific_results)} queries successfully")
if failed_queries:
    print(f"Failed queries: {len(failed_queries)} (examples: {failed_queries[:5]})")

# Manual evaluation since we can't use the standard evaluator
print("\n=== EVALUATION RESULTS (Query-Specific Corpus) ===")

k_values = [1, 3, 5, 10, 100, 1000]
metrics = {
    "precision": {f"P@{k}": 0.0 for k in k_values},
    "recall": {f"Recall@{k}": 0.0 for k in k_values},
    "ndcg": {f"NDCG@{k}": 0.0 for k in k_values},
}

# Calculate metrics manually
for query_id, results in query_specific_results.items():
    if query_id not in qrels:
        continue

    # Get relevant documents for this query
    relevant_docs = {doc_id for doc_id, score in qrels[query_id].items() if score > 0}
    if not relevant_docs:
        continue

    # Sort results by score
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for k in k_values:
        top_k_docs = [doc_id for doc_id, _ in sorted_results[:k]]

        # Precision@k
        relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        if len(top_k_docs) > 0:
            metrics["precision"][f"P@{k}"] += relevant_in_top_k / len(top_k_docs)

        # Recall@k
        if len(relevant_docs) > 0:
            metrics["recall"][f"Recall@{k}"] += relevant_in_top_k / len(relevant_docs)

        # Simplified NDCG@k (binary relevance)
        dcg = 0.0
        for i, doc_id in enumerate(top_k_docs):
            if doc_id in relevant_docs:
                dcg += 1.0 / (torch.log2(torch.tensor(i + 2.0)).item())

        # Ideal DCG
        ideal_dcg = 0.0
        for i in range(min(k, len(relevant_docs))):
            ideal_dcg += 1.0 / (torch.log2(torch.tensor(i + 2.0)).item())

        if ideal_dcg > 0:
            metrics["ndcg"][f"NDCG@{k}"] += dcg / ideal_dcg

# Average metrics
num_queries = len(query_specific_results)
if num_queries > 0:
    for metric_type in metrics:
        for k_metric in metrics[metric_type]:
            metrics[metric_type][k_metric] /= num_queries
            metrics[metric_type][k_metric] = round(metrics[metric_type][k_metric], 5)

# Print results
for metric_type, values in metrics.items():
    print(f"\n{metric_type.upper()}:")
    for k_metric, value in values.items():
        print(f"  {k_metric}: {value:.4f}")

# Print comparison with standard evaluation
print("\n=== COMPARISON: Query-Specific vs Standard Retrieval ===")
print("\nQuery-Specific Corpus Retrieval (searching only within qrels documents):")
if query_stats:
    print(f"  - Each query searches only its qrels documents")
    print(
        f"  - Average docs per query: {sum(s['total_docs_in_qrels'] for s in query_stats) / len(query_stats):.1f}"
    )
    print(
        f"  - Average relevant docs per query: {sum(s['relevant_docs'] for s in query_stats) / len(query_stats):.1f}"
    )
    print(
        f"  - Average irrelevant docs per query: {sum(s['irrelevant_docs'] for s in query_stats) / len(query_stats):.1f}"
    )

print("\nStandard BEIR Retrieval (searching entire corpus):")
print(f"  - Each query searches all {len(corpus)} corpus documents")
print(f"  - Then evaluated against qrels")

# Analyze perfect retrieval potential
print("\n=== THEORETICAL MAXIMUM PERFORMANCE ===")
print("If the model perfectly ranks all relevant docs before irrelevant ones:")

for k in [1, 3, 5, 10]:
    max_precision_sum = 0
    queries_with_enough_relevant = 0

    for stats in query_stats:
        relevant = stats["relevant_docs"]
        if relevant > 0:
            # Precision@k is min(relevant, k) / k
            max_precision_sum += min(relevant, k) / k
            if relevant >= k:
                queries_with_enough_relevant += 1

    if query_stats:
        max_avg_precision = max_precision_sum / len(query_stats)
        print(f"  Max Precision@{k}: {max_avg_precision:.3f}")
        print(
            f"    ({queries_with_enough_relevant}/{len(query_stats)} queries have ≥{k} relevant docs)"
        )

# Save results
results_dir = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

output_file = os.path.join(results_dir, "filtered-qwen.json")
with open(output_file, "w") as f:
    json.dump(
        {
            "summary": {
                "total_queries_processed": len(query_specific_results),
                "total_queries_with_qrels": len(query_stats),
                "failed_queries": len(failed_queries),
            },
            "metrics": metrics,
            "query_stats_summary": {
                "avg_docs_per_query": sum(s["total_docs_in_qrels"] for s in query_stats)
                / len(query_stats)
                if query_stats
                else 0,
                "avg_relevant_per_query": sum(s["relevant_docs"] for s in query_stats)
                / len(query_stats)
                if query_stats
                else 0,
                "avg_irrelevant_per_query": sum(
                    s["irrelevant_docs"] for s in query_stats
                )
                / len(query_stats)
                if query_stats
                else 0,
            },
        },
        f,
        indent=2,
    )

# Save detailed statistics
stats_file = os.path.join(results_dir, "query-stats.json")
with open(stats_file, "w") as f:
    json.dump(
        {
            "summary": {
                "total_queries": len(query_stats),
                "successful_queries": len(query_specific_results),
                "failed_queries": len(failed_queries),
            },
            "per_query_stats": query_stats[:50],  # Save first 50 for inspection
            "failed_query_ids": failed_queries[:20],  # Save first 20 failed
        },
        f,
        indent=2,
    )

print(f"\nResults saved to: {output_file}")
print(f"Statistics saved to: {stats_file}")

print("\n=== KEY INSIGHT ===")
print(
    "This evaluation shows performance when each query can ONLY retrieve from its qrels documents."
)
print("This is different from standard BEIR where queries search the entire corpus.")
print(
    "The results show how well the model ranks relevant vs irrelevant docs when limited to the annotated set."
)
