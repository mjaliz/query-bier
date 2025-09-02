#!/usr/bin/env python3
"""
Threshold tuning evaluation script for BEIR-style data
Calculates precision, recall, and F1 score for different similarity thresholds
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def load_beir_data(data_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load BEIR dataset (corpus, queries, qrels)"""
    corpus_path = data_path / "corpus.jsonl"
    queries_path = data_path / "query.jsonl"
    qrels_path = data_path / "qrels.tsv"

    # Load corpus
    corpus = {}
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc.get("title", "") + " " + doc.get("text", "")

    # Load queries
    queries = {}
    with open(queries_path, "r") as f:
        for line in f:
            query = json.loads(line)
            queries[query["_id"]] = query["text"]

    # Load qrels
    qrels = {}
    with open(qrels_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id, doc_id, relevance = parts[:3]
                try:
                    relevance = int(relevance)
                    if relevance > 0:  # Only consider relevant documents
                        if query_id not in qrels:
                            qrels[query_id] = {}
                        qrels[query_id][doc_id] = relevance
                except ValueError:
                    continue  # Skip lines with invalid relevance scores
    
    return corpus, queries, qrels


def compute_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute embeddings for a list of texts"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings


def calculate_metrics_at_threshold(
    similarities: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    threshold: float,
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score at a given threshold
    Treats the task as binary classification: above threshold = positive prediction
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for query_id, query_sims in similarities.items():
        relevant_docs = qrels.get(query_id, {})

        for doc_id, sim_score in query_sims.items():
            is_relevant = doc_id in relevant_docs
            is_predicted_positive = sim_score >= threshold

            if is_relevant and is_predicted_positive:
                true_positives += 1
            elif is_relevant and not is_predicted_positive:
                false_negatives += 1
            elif not is_relevant and is_predicted_positive:
                false_positives += 1
            else:
                true_negatives += 1

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    
    # Calculate accuracy with safety check for division by zero
    total = true_positives + false_positives + false_negatives + true_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def evaluate_thresholds_with_streaming(
    model_name: str,
    data_path: Path,
    thresholds: List[float],
    batch_size: int = 32,
    use_filtered_corpus: bool = True,
    progress_callback=None,
    max_queries: int = None,
) -> Dict:
    """Streaming version with detailed progress logs"""
    return evaluate_thresholds(
        model_name, data_path, thresholds, batch_size,
        use_filtered_corpus, progress_callback, max_queries
    )


def evaluate_thresholds(
    model_name: str,
    data_path: Path,
    thresholds: List[float],
    batch_size: int = 32,
    use_filtered_corpus: bool = True,
    progress_callback=None,
    max_queries: int = None,
) -> Dict:
    """
    Evaluate model at different similarity thresholds

    Args:
        model_name: Name of the embedding model
        data_path: Path to BEIR dataset
        thresholds: List of thresholds to evaluate
        batch_size: Batch size for encoding
        use_filtered_corpus: If True, only use documents in qrels for each query
        progress_callback: Optional callback for progress updates
    """

    # Load model
    if progress_callback:
        progress_callback(
            {"type": "log", "level": "info", "message": f"Loading model: {model_name}"}
        )
    model = SentenceTransformer(model_name)

    # Load data
    if progress_callback:
        progress_callback(
            {"type": "log", "level": "info", "message": "Loading BEIR data..."}
        )
    
    try:
        corpus, queries, qrels = load_beir_data(data_path)
        if progress_callback:
            progress_callback(
                {"type": "log", "level": "info", 
                 "message": f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels"}
            )
    except Exception as e:
        error_msg = f"Failed to load BEIR data from {data_path}: {str(e)}"
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)

    # Validate data
    if not corpus or not queries or not qrels:
        error_msg = "No data found in BEIR dataset. Please check the data path and file formats."
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)
    
    # Calculate similarities for all query-document pairs
    similarities = {}
    total_queries = len(qrels)
    
    if total_queries == 0:
        error_msg = "No qrels found in the dataset"
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)
    
    # Limit queries if specified
    qrels_items = list(qrels.items())
    if max_queries and max_queries < len(qrels_items):
        qrels_items = qrels_items[:max_queries]
        total_queries = max_queries
        if progress_callback:
            progress_callback({"type": "log", "level": "info", 
                             "message": f"Processing {max_queries} queries (out of {len(qrels)} total)"})
    
    # Log corpus mode details
    if progress_callback:
        if use_filtered_corpus:
            progress_callback({
                "type": "log", 
                "level": "info",
                "message": "Mode: Evaluating each query against its relevant docs + negative samples"
            })
        else:
            progress_callback({
                "type": "log", 
                "level": "info",
                "message": f"Mode: Evaluating each query against full corpus ({len(corpus)} documents)"
            })

    for idx, (query_id, relevant_docs) in enumerate(qrels_items):
        if query_id not in queries:
            continue

        query_text = queries[query_id]

        # Determine which documents to evaluate
        if use_filtered_corpus:
            # Only evaluate against documents in qrels for this query
            doc_ids = list(relevant_docs.keys())
            num_relevant = len(doc_ids)
            
            # Add some negative samples if available
            all_doc_ids = list(corpus.keys())
            num_negatives = min(len(doc_ids) * 2, 100)  # Add up to 2x negatives or 100
            negative_samples = [
                doc_id
                for doc_id in np.random.choice(
                    all_doc_ids, size=num_negatives, replace=False
                )
                if doc_id not in relevant_docs
            ]
            doc_ids.extend(negative_samples)
            
            # Log details for first query as example
            if idx == 0 and progress_callback:
                progress_callback({
                    "type": "log",
                    "level": "debug",
                    "message": f"Query {query_id}: Evaluating {num_relevant} relevant + {len(negative_samples)} negative docs"
                })
        else:
            # Evaluate against entire corpus
            doc_ids = list(corpus.keys())
            
            # Log details for first query
            if idx == 0 and progress_callback:
                progress_callback({
                    "type": "log",
                    "level": "debug", 
                    "message": f"Query {query_id}: Evaluating against all {len(doc_ids)} corpus documents"
                })

        # Get document texts
        doc_texts = [corpus[doc_id] for doc_id in doc_ids if doc_id in corpus]
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in corpus]

        if not doc_texts:
            continue

        # Compute embeddings
        query_embedding = model.encode([query_text], convert_to_numpy=True)
        doc_embeddings = model.encode(
            doc_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Calculate cosine similarities
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )
        doc_embeddings = doc_embeddings / np.linalg.norm(
            doc_embeddings, axis=1, keepdims=True
        )
        sims = np.dot(doc_embeddings, query_embedding.T).flatten()

        # Store similarities
        similarities[query_id] = {
            doc_id: float(sim) for doc_id, sim in zip(doc_ids, sims)
        }

        # Progress update - more frequent updates
        if progress_callback:
            if idx % 5 == 0 or idx == total_queries - 1:  # Update every 5 queries or on last
                progress = int(
                    (idx + 1) / total_queries * 50
                )  # First 50% for similarity calculation
                progress_callback(
                    {
                        "type": "progress",
                        "progress": progress,
                        "message": f"Computing similarities: {idx + 1}/{total_queries} queries",
                    }
                )
                
                # Log memory usage periodically
                if idx % 20 == 0 and idx > 0:
                    num_pairs_evaluated = sum(len(sims) for sims in similarities.values())
                    progress_callback({
                        "type": "log",
                        "level": "debug",
                        "message": f"Evaluated {num_pairs_evaluated} query-document pairs so far"
                    })

    # Check if we have any similarities to evaluate
    if not similarities:
        error_msg = "No similarities computed. Please check if queries and documents are matching."
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)
    
    # Log summary of similarities computed
    total_pairs = sum(len(sims) for sims in similarities.values())
    if progress_callback:
        progress_callback({
            "type": "log",
            "level": "info",
            "message": f"Computed {total_pairs} query-document similarity scores"
        })
        progress_callback({
            "type": "log",
            "level": "info",
            "message": f"Starting threshold evaluation for {len(thresholds)} thresholds: {thresholds}"
        })
    
    # Evaluate at different thresholds
    results = []
    for i, threshold in enumerate(thresholds):
        metrics = calculate_metrics_at_threshold(similarities, qrels, threshold)
        results.append(metrics)

        if progress_callback:
            progress = 50 + int(
                (i + 1) / len(thresholds) * 50
            )  # Last 50% for threshold evaluation
            progress_callback(
                {
                    "type": "progress",
                    "progress": progress,
                    "message": f"Evaluating threshold {threshold:.3f}",
                }
            )
            
            # Log metrics for this threshold
            progress_callback({
                "type": "log",
                "level": "debug",
                "message": f"Threshold {threshold:.3f}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
            })

    # Find best threshold by F1 score
    if not results:
        error_msg = "No results generated from threshold evaluation"
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)
    
    best_result = max(results, key=lambda x: x["f1"])
    
    # Log final summary
    if progress_callback:
        progress_callback({
            "type": "log",
            "level": "info",
            "message": f"✓ Evaluation complete! Best threshold: {best_result['threshold']:.3f} (F1={best_result['f1']:.3f})"
        })

    return {
        "model_name": model_name,
        "thresholds_evaluated": thresholds,
        "results": results,
        "best_threshold": best_result["threshold"],
        "best_f1": best_result["f1"],
        "best_precision": best_result["precision"],
        "best_recall": best_result["recall"],
        "corpus_size": len(corpus),
        "queries_evaluated": len(similarities),
        "use_filtered_corpus": use_filtered_corpus,
    }


def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate similarity thresholds for BEIR data"
    )
    parser.add_argument(
        "--model-name", required=True, help="Name of the embedding model"
    )
    parser.add_argument("--data-path", default="scifact", help="Path to BEIR dataset")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Thresholds to evaluate (default: 0.1 to 0.9 step 0.1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for encoding"
    )
    parser.add_argument(
        "--use-filtered-corpus",
        action="store_true",
        help="Only evaluate on documents in qrels",
    )
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Default thresholds if not specified
    if args.thresholds is None:
        args.thresholds = [i / 10 for i in range(1, 10)]  # 0.1 to 0.9

    # Run evaluation
    results = evaluate_thresholds(
        model_name=args.model_name,
        data_path=Path(args.data_path),
        thresholds=args.thresholds,
        batch_size=args.batch_size,
        use_filtered_corpus=args.use_filtered_corpus,
    )

    # Save or print results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

    # Print summary
    print(f"\nBest threshold: {results['best_threshold']:.3f}")
    print(f"Best F1 score: {results['best_f1']:.3f}")
    print(f"Precision at best: {results['best_precision']:.3f}")
    print(f"Recall at best: {results['best_recall']:.3f}")


if __name__ == "__main__":
    main()
