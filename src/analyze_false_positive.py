#!/usr/bin/env python3
"""
Analyze false positives: documents that appear in top-10 results but are not relevant
"""

import json
import logging
from pathlib import Path

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
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

# Load the model and perform retrieval
print("\nPerforming retrieval...")
model = DRES(
    models.SentenceBERT(
        "BAAI/bge-m3",
        trust_remote_code=True,
    ),
    batch_size=16,
)

retriever = EvaluateRetrieval(model, score_function="cos_sim")
results = retriever.retrieve(corpus, queries)

# Analyze false positives in top-10
print("\n=== ANALYZING FALSE POSITIVES IN TOP-10 RESULTS ===\n")

k = 10
false_positives_report = []
total_false_positives = 0
total_true_positives = 0

for query_id in queries.keys():
    # Get top-k results for this query
    query_results = results.get(query_id, {})
    top_k_docs = sorted(query_results.items(), key=lambda x: x[1], reverse=True)[:k]

    # Get relevant documents from qrels
    relevant_docs = {
        doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0
    }

    # Find false positives (retrieved but not relevant)
    false_positives = []
    true_positives = []

    for doc_id, score in top_k_docs:
        if doc_id in relevant_docs:
            true_positives.append((doc_id, score))
            total_true_positives += 1
        else:
            false_positives.append((doc_id, score))
            total_false_positives += 1

    # Store detailed information
    if false_positives:  # Only report queries with false positives
        report_entry = {
            "query_id": query_id,
            "query_text": queries[query_id],
            "total_relevant_docs": len(relevant_docs),
            "precision_at_10": len(true_positives) / k,
            "true_positives": {
                "count": len(true_positives),
                "docs": [
                    {
                        "doc_id": doc_id,
                        "score": round(score, 4),
                        "title": corpus[doc_id].get("title", ""),
                        "text_preview": corpus[doc_id].get("text", "")[:100] + "...",
                    }
                    for doc_id, score in true_positives[:3]  # Show first 3
                ],
            },
            "false_positives": {
                "count": len(false_positives),
                "docs": [
                    {
                        "doc_id": doc_id,
                        "score": round(score, 4),
                        "title": corpus[doc_id].get("title", ""),
                        "text_preview": corpus[doc_id].get("text", "")[:100] + "...",
                    }
                    for doc_id, score in false_positives[:5]  # Show first 5
                ],
            },
        }
        false_positives_report.append(report_entry)

# Sort by number of false positives (worst first)
false_positives_report.sort(key=lambda x: x["false_positives"]["count"], reverse=True)

# Print summary statistics
print(f"SUMMARY STATISTICS:")
print(f"Total queries analyzed: {len(queries)}")
print(f"Total false positives in top-10: {total_false_positives}")
print(f"Total true positives in top-10: {total_true_positives}")
print(f"Average false positives per query: {total_false_positives / len(queries):.2f}")
print(f"Average precision@10: {total_true_positives / (len(queries) * k):.3f}")

# Show worst offenders
print("\n=== QUERIES WITH MOST FALSE POSITIVES ===\n")
for i, entry in enumerate(false_positives_report[:5], 1):
    print(f"{i}. Query ID: {entry['query_id']}")
    print(f'   Query: "{entry["query_text"][:80]}..."')
    print(f"   Precision@10: {entry['precision_at_10']:.1%}")
    print(f"   False positives: {entry['false_positives']['count']}/{k}")
    print(f"   True positives: {entry['true_positives']['count']}/{k}")

    print(f"\n   TOP FALSE POSITIVE DOCUMENTS (not relevant but retrieved):")
    for j, doc in enumerate(entry["false_positives"]["docs"][:3], 1):
        print(f"   {j}. Doc ID: {doc['doc_id']} (score: {doc['score']})")
        print(f"      Title: {doc['title'][:60] if doc['title'] else 'N/A'}")
        print(f"      Text: {doc['text_preview'][:80]}")

    if entry["true_positives"]["docs"]:
        print(f"\n   TOP TRUE POSITIVE DOCUMENTS (relevant and retrieved):")
        for j, doc in enumerate(entry["true_positives"]["docs"][:2], 1):
            print(f"   {j}. Doc ID: {doc['doc_id']} (score: {doc['score']})")
            print(f"      Title: {doc['title'][:60] if doc['title'] else 'N/A'}")
            print(f"      Text: {doc['text_preview'][:80]}")
    print("-" * 80)

# Save full report to JSON
output_file = Path(__file__).parent / "false_positives_analysis.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "summary": {
                "total_queries": len(queries),
                "total_false_positives": total_false_positives,
                "total_true_positives": total_true_positives,
                "avg_false_positives_per_query": total_false_positives / len(queries),
                "avg_precision_at_10": total_true_positives / (len(queries) * k),
            },
            "detailed_report": false_positives_report[:20],  # Save top 20 worst queries
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"\nFull report saved to: {output_file}")

# Additional analysis: Find most common false positive documents
print("\n=== MOST COMMON FALSE POSITIVE DOCUMENTS ===")
false_positive_doc_counts = {}
for entry in false_positives_report:
    for doc in entry["false_positives"]["docs"]:
        doc_id = doc["doc_id"]
        if doc_id not in false_positive_doc_counts:
            false_positive_doc_counts[doc_id] = {
                "count": 0,
                "queries": [],
                "title": doc["title"],
                "text_preview": doc["text_preview"],
            }
        false_positive_doc_counts[doc_id]["count"] += 1
        false_positive_doc_counts[doc_id]["queries"].append(entry["query_id"])

# Sort by frequency
sorted_fp_docs = sorted(
    false_positive_doc_counts.items(), key=lambda x: x[1]["count"], reverse=True
)

print("\nDocuments that appear as false positives for multiple queries:")
for doc_id, info in sorted_fp_docs[:5]:
    if info["count"] > 1:
        print(f"\nDoc ID: {doc_id}")
        print(f"  Appears as false positive for {info['count']} queries")
        print(f"  Title: {info['title'][:60] if info['title'] else 'N/A'}")
        print(f"  Text: {info['text_preview'][:100]}")
        print(f"  Query IDs: {', '.join(info['queries'][:5])}...")
