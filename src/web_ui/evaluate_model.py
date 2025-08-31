#!/usr/bin/env python3
"""
Evaluation script for embedding models that can be called by the web UI
Supports both standard BEIR evaluation and query-specific corpus filtering
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


class EvaluationProgress:
    def __init__(self):
        self.current = 0
        self.total = 100

    def update(self, current, total=None):
        self.current = current
        if total:
            self.total = total
        progress = int((self.current / self.total) * 100)
        msg = json.dumps(
            {
                "type": "progress",
                "progress": progress,
                "message": f"Processing {self.current}/{self.total}",
            }
        )
        print(msg, flush=True)
        sys.stdout.flush()  # Force flush

    def log(self, message, level="info"):
        msg = json.dumps({"type": "log", "level": level, "message": str(message)})
        print(msg, flush=True)
        sys.stdout.flush()  # Force flush


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


def evaluate_model(
    model_name: str,
    output_name: Optional[str] = None,
    batch_size: int = 32,
    use_filtered_corpus: bool = True,
):
    """
    Evaluate an embedding model on the BEIR dataset with query-specific corpus filtering

    Args:
        model_name: HuggingFace model name or local path
        output_name: Output file name (without extension)
        batch_size: Batch size for evaluation
        use_filtered_corpus: Whether to use query-specific corpus filtering
    """
    progress = EvaluationProgress()

    try:
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[LoggingHandler()],
        )

        # Determine data directory
        DATA_DIR = Path(__file__).parent.parent.parent / "data" / "beir_data"

        evaluation_mode = (
            "query-specific corpus" if use_filtered_corpus else "full corpus"
        )
        progress.log(f"Using {evaluation_mode} evaluation mode")
        progress.update(10)

        # Load data
        corpus_path = DATA_DIR / "corpus.jsonl"
        query_path = DATA_DIR / "query.jsonl"
        qrels_path = DATA_DIR / "qrels.tsv"

        progress.log("Loading corpus, queries, and qrels...")
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(corpus_path),
            query_file=str(query_path),
            qrels_file=str(qrels_path),
        ).load_custom()

        progress.log(f"Loaded {len(corpus)} documents and {len(queries)} queries")
        progress.update(20)

        # Load model
        progress.log(f"Loading model: {model_name}")

        # Try to load as SentenceTransformer first
        try:
            model = DRES(models.SentenceBERT(model_name), batch_size=batch_size)
            progress.log("Model loaded as SentenceTransformer")
        except Exception as e:
            # Try with trust_remote_code for models like Jina
            progress.log("Trying to load with trust_remote_code=True")
            model = DRES(
                models.SentenceBERT(
                    model_name,
                    trust_remote_code=True,
                    model_kwargs={"default_task": "retrieval.query"},
                ),
                batch_size=batch_size,
            )
            progress.log("Model loaded with trust_remote_code")

        progress.update(40)

        if use_filtered_corpus:
            # Query-specific corpus evaluation
            progress.log("Running query-specific corpus evaluation...")
            progress.log("Each query will ONLY search within its qrels documents")
            progress.update(50)

            query_specific_results = {}
            query_stats = []
            failed_queries = []

            query_ids = list(queries.keys())
            total_queries = len(query_ids)

            for i, query_id in enumerate(query_ids, 1):
                if i % 100 == 0:
                    progress.update(50 + int((i / total_queries) * 30))
                    progress.log(f"Processing query {i}/{total_queries}...")

                query_text = queries[query_id]

                # Get ALL documents in qrels for this query (both relevant and irrelevant)
                query_docs_in_qrels = qrels.get(query_id, {})

                if not query_docs_in_qrels:
                    continue

                # Filter corpus to only documents in qrels for this query
                query_specific_corpus = {
                    doc_id: corpus[doc_id]
                    for doc_id in query_docs_in_qrels.keys()
                    if doc_id in corpus
                }

                # Count relevant vs irrelevant
                relevant_count = sum(
                    1 for score in query_docs_in_qrels.values() if score > 0
                )
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
                    continue

                try:
                    # Use custom search function
                    results = custom_search_single_query(
                        model, query_text, query_specific_corpus, top_k=1000
                    )
                    query_specific_results[query_id] = results
                except Exception as e:
                    failed_queries.append(query_id)
                    continue

            progress.update(80)
            results = query_specific_results

            # Manual evaluation for query-specific corpus
            progress.log("Calculating metrics...")
            k_values = [1, 3, 5, 10, 100, 1000]
            precision = {k: 0.0 for k in k_values}
            recall = {k: 0.0 for k in k_values}
            ndcg = {k: 0.0 for k in k_values}

            # Calculate metrics manually
            for query_id, query_results in results.items():
                if query_id not in qrels:
                    continue

                # Get relevant documents for this query
                relevant_docs = {
                    doc_id for doc_id, score in qrels[query_id].items() if score > 0
                }
                if not relevant_docs:
                    continue

                # Sort results by score
                sorted_results = sorted(
                    query_results.items(), key=lambda x: x[1], reverse=True
                )

                for k in k_values:
                    top_k_docs = [doc_id for doc_id, _ in sorted_results[:k]]

                    # Precision@k
                    relevant_in_top_k = sum(
                        1 for doc_id in top_k_docs if doc_id in relevant_docs
                    )
                    if len(top_k_docs) > 0:
                        precision[k] += relevant_in_top_k / len(top_k_docs)

                    # Recall@k
                    if len(relevant_docs) > 0:
                        recall[k] += relevant_in_top_k / len(relevant_docs)

                    # Simplified NDCG@k (binary relevance)
                    dcg = 0.0
                    for idx, doc_id in enumerate(top_k_docs):
                        if doc_id in relevant_docs:
                            dcg += 1.0 / (torch.log2(torch.tensor(idx + 2.0)).item())

                    # Ideal DCG
                    ideal_dcg = 0.0
                    for idx in range(min(k, len(relevant_docs))):
                        ideal_dcg += 1.0 / (torch.log2(torch.tensor(idx + 2.0)).item())

                    if ideal_dcg > 0:
                        ndcg[k] += dcg / ideal_dcg

            # Average metrics
            num_queries = len(results)
            if num_queries > 0:
                for k in k_values:
                    precision[k] /= num_queries
                    recall[k] /= num_queries
                    ndcg[k] /= num_queries

            _map = {}  # MAP not calculated for query-specific

        else:
            # Standard BEIR evaluation
            progress.log("Running standard BEIR evaluation...")
            retriever = EvaluateRetrieval(model, score_function="cos_sim")

            progress.log("Running retrieval (this may take a while)...")
            progress.update(50)

            # Split queries for progress tracking
            query_ids = list(queries.keys())
            total_queries = len(query_ids)
            batch_size_queries = (
                min(100, total_queries // 10) if total_queries > 100 else total_queries
            )

            results = {}
            for i in range(0, total_queries, batch_size_queries):
                batch_queries = {
                    qid: queries[qid] for qid in query_ids[i : i + batch_size_queries]
                }
                batch_results = retriever.retrieve(corpus, batch_queries)
                results.update(batch_results)

                current_progress = 50 + int((i / total_queries) * 30)
                progress.update(current_progress)
                progress.log(
                    f"Processed {min(i + batch_size_queries, total_queries)}/{total_queries} queries"
                )

            progress.update(80)

            # Evaluate
            progress.log("Evaluating metrics...")
            ndcg, _map, recall, precision = retriever.evaluate(
                qrels, results, retriever.k_values
            )
            query_stats = []

        # Calculate additional statistics
        total_queries_processed = len(results)
        total_queries_with_qrels = len([q for q in results if q in qrels])

        if use_filtered_corpus and query_stats:
            # Use query-specific stats
            avg_docs_per_query = sum(
                s["total_docs_in_qrels"] for s in query_stats
            ) / len(query_stats)
            avg_relevant_per_query = sum(s["relevant_docs"] for s in query_stats) / len(
                query_stats
            )
            avg_irrelevant_per_query = sum(
                s["irrelevant_docs"] for s in query_stats
            ) / len(query_stats)
        else:
            # Calculate from results
            docs_per_query = []
            relevant_per_query = []
            for qid in results:
                if qid in qrels:
                    docs_per_query.append(len(results[qid]))
                    relevant_per_query.append(len(qrels[qid]))

            avg_docs_per_query = np.mean(docs_per_query) if docs_per_query else 0
            avg_relevant_per_query = (
                np.mean(relevant_per_query) if relevant_per_query else 0
            )
            avg_irrelevant_per_query = avg_docs_per_query - avg_relevant_per_query

        progress.update(90)

        # Prepare output
        output_data = {
            "summary": {
                "total_queries_processed": total_queries_processed,
                "total_queries_with_qrels": total_queries_with_qrels,
                "failed_queries": len(failed_queries) if use_filtered_corpus else 0,
                "model_name": model_name,
                "evaluation_mode": "query-specific corpus"
                if use_filtered_corpus
                else "standard BEIR",
            },
            "metrics": {
                "precision": {f"P@{k}": v for k, v in precision.items()},
                "recall": {f"Recall@{k}": v for k, v in recall.items()},
                "ndcg": {f"NDCG@{k}": v for k, v in ndcg.items()},
                "map": {f"MAP@{k}": v for k, v in _map.items()} if _map else {},
            },
            "query_stats_summary": {
                "avg_docs_per_query": avg_docs_per_query,
                "avg_relevant_per_query": avg_relevant_per_query,
                "avg_irrelevant_per_query": avg_irrelevant_per_query,
            },
        }

        # Save results
        results_dir = Path(__file__).parent.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)

        if not output_name:
            # Generate output name from model name
            output_name = model_name.replace("/", "-").replace("\\", "-")

        output_file = results_dir / f"{output_name}.json"

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        progress.log(f"Results saved to: {output_file}")
        progress.update(100)

        # Send completion message
        print(
            json.dumps(
                {
                    "type": "complete",
                    "output_file": str(output_file.name),
                    "metrics": {
                        "P@10": precision.get(10, 0),
                        "R@10": recall.get(10, 0),
                        "NDCG@10": ndcg.get(10, 0),
                    },
                }
            ),
            flush=True,
        )

    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an embedding model")
    parser.add_argument("--model-name", required=True, help="Model name or path")
    parser.add_argument("--output-name", help="Output file name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--use-filtered-corpus", action="store_true", help="Use filtered corpus"
    )

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model_name,
        output_name=args.output_name,
        batch_size=args.batch_size,
        use_filtered_corpus=args.use_filtered_corpus,
    )
