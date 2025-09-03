#!/usr/bin/env python3
"""
Threshold tuning evaluation script for BEIR-style data
Calculates precision, recall, and F1 score for different similarity thresholds
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    Computes metrics per query and then averages across all queries
    """
    query_metrics = []
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_true_negatives = 0

    for query_id, query_sims in similarities.items():
        relevant_docs = qrels.get(query_id, {})
        
        if not relevant_docs:
            # Skip queries with no relevant documents
            continue
        
        query_tp = 0
        query_fp = 0
        query_fn = 0
        query_tn = 0

        for doc_id, sim_score in query_sims.items():
            is_relevant = doc_id in relevant_docs
            is_predicted_positive = sim_score >= threshold

            if is_relevant and is_predicted_positive:
                query_tp += 1
            elif is_relevant and not is_predicted_positive:
                query_fn += 1
            elif not is_relevant and is_predicted_positive:
                query_fp += 1
            else:
                query_tn += 1

        # Calculate per-query metrics
        query_precision = (
            query_tp / (query_tp + query_fp)
            if (query_tp + query_fp) > 0
            else 0
        )
        query_recall = (
            query_tp / (query_tp + query_fn)
            if (query_tp + query_fn) > 0
            else 0
        )
        query_f1 = (
            2 * (query_precision * query_recall) / (query_precision + query_recall)
            if (query_precision + query_recall) > 0
            else 0
        )
        
        query_metrics.append({
            "precision": query_precision,
            "recall": query_recall,
            "f1": query_f1
        })
        
        # Accumulate for total counts
        total_true_positives += query_tp
        total_false_positives += query_fp
        total_false_negatives += query_fn
        total_true_negatives += query_tn

    # Calculate averaged metrics across all queries
    if query_metrics:
        avg_precision = float(np.mean([m["precision"] for m in query_metrics]))
        avg_recall = float(np.mean([m["recall"] for m in query_metrics]))
        avg_f1 = float(np.mean([m["f1"] for m in query_metrics]))
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
    
    # Calculate overall accuracy
    total = total_true_positives + total_false_positives + total_false_negatives + total_true_negatives
    accuracy = (total_true_positives + total_true_negatives) / total if total > 0 else 0

    return {
        "threshold": threshold,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "accuracy": accuracy,
        "true_positives": total_true_positives,
        "false_positives": total_false_positives,
        "false_negatives": total_false_negatives,
        "true_negatives": total_true_negatives,
        "num_queries_evaluated": len(query_metrics),
    }


def evaluate_thresholds_with_streaming(
    model_name: str,
    data_path: Path,
    thresholds: List[float],
    batch_size: int = 32,
    use_filtered_corpus: bool = True,
    progress_callback=None,
    max_queries: Optional[int] = None,
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
    max_queries: Optional[int] = None,
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

    # Check for GPU availability (CUDA or MPS)
    import torch
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        device_msg = f"🚀 CUDA available! Using GPU: {gpu_name}"
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_msg = "🚀 MPS (Metal) available! Using Apple Silicon GPU"
    else:
        device = 'cpu'
        device_msg = "⚠️ No GPU available, using CPU (this will be slower)"
    
    if progress_callback:
        level = "info" if device != 'cpu' else "warning"
        progress_callback(
            {"type": "log", "level": level, "message": device_msg}
        )
    
    # Load model
    if progress_callback:
        progress_callback(
            {"type": "log", "level": "info", "message": f"Loading model: {model_name}"}
        )
    
    # SentenceTransformers doesn't support MPS yet, fall back to CPU for MPS
    if device == 'mps':
        model_device = 'cpu'
        if progress_callback:
            progress_callback(
                {"type": "log", "level": "info", 
                 "message": "Note: SentenceTransformers doesn't support MPS yet, using CPU"}
            )
    else:
        model_device = device
    
    try:
        # Explicitly set device for the model
        model = SentenceTransformer(model_name, device=model_device)
        if progress_callback:
            progress_callback(
                {"type": "log", "level": "info", 
                 "message": f"✓ Model loaded successfully on {model_device.upper()}"}
            )
    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)

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
    
    # Pre-compute corpus embeddings if using full corpus mode
    corpus_embeddings_dict = {}
    
    if not use_filtered_corpus:
        if progress_callback:
            progress_callback({
                "type": "log", 
                "level": "info",
                "message": f"Pre-computing embeddings for {len(corpus)} corpus documents..."
            })
        
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
        
        # Use larger batch size for GPU if available
        effective_batch_size = batch_size * 4 if model_device == 'cuda' else batch_size
        
        # Compute all corpus embeddings once
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=effective_batch_size,
            convert_to_numpy=True,
            show_progress_bar=progress_callback is not None,
        )
        
        # Normalize embeddings for cosine similarity
        corpus_embeddings = corpus_embeddings / np.linalg.norm(
            corpus_embeddings, axis=1, keepdims=True
        )
        
        # Store in dictionary for fast lookup
        for doc_id, embedding in zip(corpus_ids, corpus_embeddings):
            corpus_embeddings_dict[doc_id] = embedding
        
        if progress_callback:
            progress_callback({
                "type": "log",
                "level": "info", 
                "message": f"✓ Pre-computed {len(corpus)} corpus embeddings"
            })
    
    # Calculate similarities for all query-document pairs
    similarities = {}
    total_queries = len(qrels)
    start_time = time.time()
    
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
    else:
        if progress_callback:
            progress_callback({"type": "log", "level": "info", 
                             "message": f"Processing ALL {len(qrels)} queries in the dataset"})
    
    # Log corpus mode details and estimate computations
    if progress_callback:
        if use_filtered_corpus:
            # Estimate average docs per query for filtered mode
            avg_relevant = sum(len(docs) for docs in qrels_items[:min(10, len(qrels_items))]) / min(10, len(qrels_items))
            avg_negatives = min(avg_relevant * 2, 100)
            estimated_docs_per_query = avg_relevant + avg_negatives
            total_estimated_computations = int(total_queries * estimated_docs_per_query)
            
            progress_callback({
                "type": "log", 
                "level": "info",
                "message": f"Mode: Filtered Corpus - ~{int(estimated_docs_per_query)} docs per query"
            })
            progress_callback({
                "type": "log",
                "level": "info", 
                "message": f"Estimated computations: {total_estimated_computations:,} similarity scores"
            })
        else:
            total_computations = total_queries * len(corpus)
            progress_callback({
                "type": "log", 
                "level": "info",
                "message": f"Mode: Full Corpus - {len(corpus):,} docs per query"
            })
            progress_callback({
                "type": "log",
                "level": "warning",
                "message": f"⚠️ Warning: {total_computations:,} computations needed (very slow!)"
            })
            
            # Estimate time
            sims_per_sec = 50000 if model_device == 'cuda' else 5000  # Rough estimate
            estimated_time = total_computations / sims_per_sec
            if estimated_time > 60:
                progress_callback({
                    "type": "log",
                    "level": "warning", 
                    "message": f"Estimated time: ~{estimated_time/60:.1f} minutes on {model_device.upper()}"
                })
            else:
                progress_callback({
                    "type": "log",
                    "level": "info",
                    "message": f"Estimated time: ~{estimated_time:.0f} seconds on {model_device.upper()}"
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

        # Compute query embedding
        query_embedding = model.encode(
            [query_text], 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )
        
        # Get or compute document embeddings
        if use_filtered_corpus:
            # Compute embeddings only for the selected documents
            effective_batch_size = batch_size * 4 if model_device == 'cuda' else batch_size
            
            doc_embeddings = model.encode(
                doc_texts,
                batch_size=effective_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            
            # Normalize document embeddings
            doc_embeddings = doc_embeddings / np.linalg.norm(
                doc_embeddings, axis=1, keepdims=True
            )
        else:
            # Use pre-computed embeddings
            doc_embeddings = np.array([corpus_embeddings_dict[doc_id] for doc_id in doc_ids])
        
        # Calculate cosine similarities
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
                
                # Log speed and memory usage periodically
                if idx % 20 == 0 and idx > 0:
                    num_pairs_evaluated = sum(len(sims) for sims in similarities.values())
                    elapsed = time.time() - start_time
                    speed = idx / elapsed
                    progress_callback({
                        "type": "log",
                        "level": "debug",
                        "message": f"Speed: {speed:.1f} queries/sec | Total pairs: {num_pairs_evaluated}"
                    })

    # Check if we have any similarities to evaluate
    if not similarities:
        error_msg = "No similarities computed. Please check if queries and documents are matching."
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)
    
    # Log summary of similarities computed with timing
    total_pairs = sum(len(sims) for sims in similarities.values())
    encoding_time = time.time() - start_time
    
    if progress_callback:
        progress_callback({
            "type": "log",
            "level": "info",
            "message": f"✓ Computed {total_pairs:,} similarity scores in {encoding_time:.1f}s"
        })
        
        avg_speed = total_pairs / encoding_time if encoding_time > 0 else 0
        progress_callback({
            "type": "log", 
            "level": "info",
            "message": f"Average encoding speed: {avg_speed:,.0f} pairs/sec on {model_device.upper()}"
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


def evaluate_fixed_threshold_with_details(
    model_name: str,
    data_path: Path,
    threshold: float,
    batch_size: int = 32,
    max_queries: Optional[int] = None,
    max_false_positives_per_query: int = 10,
    progress_callback=None,
) -> Dict:
    """
    Evaluate model at a fixed threshold with detailed analysis including false positives
    
    Args:
        model_name: Name of the embedding model
        data_path: Path to BEIR dataset
        threshold: Fixed threshold to evaluate
        batch_size: Batch size for encoding
        max_queries: Maximum number of queries to process
        max_false_positives_per_query: Maximum number of false positives to return per query
        progress_callback: Optional callback for progress updates
    """
    
    # Check for GPU availability (CUDA or MPS)
    import torch
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        device_msg = f"🚀 CUDA available! Using GPU: {gpu_name}"
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_msg = "🚀 MPS (Metal) available! Using Apple Silicon GPU"
    else:
        device = 'cpu'
        device_msg = "⚠️ No GPU available, using CPU (this will be slower)"
    
    if progress_callback:
        level = "info" if device != 'cpu' else "warning"
        progress_callback(
            {"type": "log", "level": level, "message": device_msg}
        )
    
    # Load model
    if progress_callback:
        progress_callback(
            {"type": "log", "level": "info", "message": f"Loading model: {model_name}"}
        )
    
    # SentenceTransformers doesn't support MPS yet, fall back to CPU for MPS
    if device == 'mps':
        model_device = 'cpu'
        if progress_callback:
            progress_callback(
                {"type": "log", "level": "info", 
                 "message": "Note: SentenceTransformers doesn't support MPS yet, using CPU"}
            )
    else:
        model_device = device
    
    try:
        model = SentenceTransformer(model_name, device=model_device)
        if progress_callback:
            progress_callback(
                {"type": "log", "level": "info", 
                 "message": f"✓ Model loaded successfully on {model_device.upper()}"}
            )
    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        if progress_callback:
            progress_callback({"type": "error", "message": error_msg})
        raise Exception(error_msg)

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
    
    # Pre-compute corpus embeddings
    if progress_callback:
        progress_callback({
            "type": "log", 
            "level": "info",
            "message": f"Pre-computing embeddings for {len(corpus)} corpus documents..."
        })
    
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
    
    # Use larger batch size for GPU if available
    effective_batch_size = batch_size * 4 if model_device == 'cuda' else batch_size
    
    # Compute all corpus embeddings once
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=effective_batch_size,
        convert_to_numpy=True,
        show_progress_bar=progress_callback is not None,
    )
    
    # Normalize embeddings for cosine similarity
    corpus_embeddings = corpus_embeddings / np.linalg.norm(
        corpus_embeddings, axis=1, keepdims=True
    )
    
    # Store in dictionary for fast lookup
    corpus_embeddings_dict = {}
    for doc_id, embedding in zip(corpus_ids, corpus_embeddings):
        corpus_embeddings_dict[doc_id] = embedding
    
    if progress_callback:
        progress_callback({
            "type": "log",
            "level": "info", 
            "message": f"✓ Pre-computed {len(corpus)} corpus embeddings"
        })
    
    # Process queries
    qrels_items = list(qrels.items())
    if max_queries and max_queries < len(qrels_items):
        qrels_items = qrels_items[:max_queries]
    
    total_queries = len(qrels_items)
    
    if progress_callback:
        progress_callback({"type": "log", "level": "info", 
                         "message": f"Processing {total_queries} queries at threshold {threshold:.3f}"})
    
    # Detailed results storage
    query_details = []
    overall_metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
    }
    
    start_time = time.time()
    
    for idx, (query_id, relevant_docs) in enumerate(qrels_items):
        if query_id not in queries:
            continue
        
        query_text = queries[query_id]
        
        # Compute query embedding
        query_embedding = model.encode(
            [query_text], 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )
        
        # Compute similarities with all corpus documents
        doc_embeddings = np.array([corpus_embeddings_dict[doc_id] for doc_id in corpus_ids])
        sims = np.dot(doc_embeddings, query_embedding.T).flatten()
        
        # Create similarity dictionary
        doc_similarities = {doc_id: float(sim) for doc_id, sim in zip(corpus_ids, sims)}
        
        # Analyze results at the threshold
        true_positives = []
        false_positives = []
        false_negatives = []
        
        for doc_id, sim_score in doc_similarities.items():
            is_relevant = doc_id in relevant_docs
            is_above_threshold = sim_score >= threshold
            
            if is_relevant and is_above_threshold:
                true_positives.append({
                    "doc_id": doc_id,
                    "score": sim_score,
                    "text": corpus[doc_id][:150],  # First 150 chars
                    "relevance": relevant_docs[doc_id]
                })
                overall_metrics["true_positives"] += 1
            elif is_relevant and not is_above_threshold:
                false_negatives.append({
                    "doc_id": doc_id,
                    "score": sim_score,
                    "text": corpus[doc_id][:150],
                    "relevance": relevant_docs[doc_id]
                })
                overall_metrics["false_negatives"] += 1
            elif not is_relevant and is_above_threshold:
                # Find which query this document actually belongs to (if any)
                actual_relevant_queries = []
                for other_query_id, other_relevant_docs in qrels.items():
                    if doc_id in other_relevant_docs:
                        actual_relevant_queries.append({
                            "query_id": other_query_id,
                            "query_text": queries.get(other_query_id, "Unknown query"),
                            "relevance": other_relevant_docs[doc_id]
                        })
                
                false_positives.append({
                    "doc_id": doc_id,
                    "score": sim_score,
                    "text": corpus[doc_id][:150],
                    "actual_relevant_queries": actual_relevant_queries  # This shows what queries this doc SHOULD match
                })
                overall_metrics["false_positives"] += 1
            else:
                overall_metrics["true_negatives"] += 1
        
        # Sort false positives by score (highest first) and limit
        false_positives.sort(key=lambda x: x["score"], reverse=True)
        false_positives = false_positives[:max_false_positives_per_query]
        
        # Calculate per-query metrics
        query_tp = len(true_positives)
        query_fp = len([1 for doc_id, sim in doc_similarities.items() 
                        if doc_id not in relevant_docs and sim >= threshold])
        query_fn = len(false_negatives)
        
        query_precision = query_tp / (query_tp + query_fp) if (query_tp + query_fp) > 0 else 0
        query_recall = query_tp / (query_tp + query_fn) if (query_tp + query_fn) > 0 else 0
        query_f1 = 2 * (query_precision * query_recall) / (query_precision + query_recall) if (query_precision + query_recall) > 0 else 0
        
        query_details.append({
            "query_id": query_id,
            "query_text": query_text,
            "metrics": {
                "precision": query_precision,
                "recall": query_recall,
                "f1": query_f1,
                "true_positives_count": query_tp,
                "false_positives_count": query_fp,
                "false_negatives_count": query_fn
            },
            "true_positives": true_positives,
            "false_positives": false_positives,  # Limited to max_false_positives_per_query
            "false_negatives": false_negatives
        })
        
        # Progress update
        if progress_callback and (idx % 10 == 0 or idx == total_queries - 1):
            progress = int((idx + 1) / total_queries * 100)
            progress_callback({
                "type": "progress",
                "progress": progress,
                "message": f"Processing query {idx + 1}/{total_queries}"
            })
    
    # Calculate overall metrics
    overall_precision = (
        overall_metrics["true_positives"] / 
        (overall_metrics["true_positives"] + overall_metrics["false_positives"])
        if (overall_metrics["true_positives"] + overall_metrics["false_positives"]) > 0
        else 0
    )
    overall_recall = (
        overall_metrics["true_positives"] / 
        (overall_metrics["true_positives"] + overall_metrics["false_negatives"])
        if (overall_metrics["true_positives"] + overall_metrics["false_negatives"]) > 0
        else 0
    )
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )
    
    # Calculate average metrics
    if query_details:
        avg_precision = np.mean([q["metrics"]["precision"] for q in query_details])
        avg_recall = np.mean([q["metrics"]["recall"] for q in query_details])
        avg_f1 = np.mean([q["metrics"]["f1"] for q in query_details])
    else:
        avg_precision = avg_recall = avg_f1 = 0
    
    encoding_time = time.time() - start_time
    
    if progress_callback:
        progress_callback({
            "type": "log",
            "level": "info",
            "message": f"✓ Evaluation complete in {encoding_time:.1f}s"
        })
    
    # Send summary first
    summary_result = {
        "model_name": model_name,
        "threshold": threshold,
        "overall_metrics": {
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1),
            "accuracy": float(
                (overall_metrics["true_positives"] + overall_metrics["true_negatives"]) /
                sum(overall_metrics.values()) if sum(overall_metrics.values()) > 0 else 0
            ),
            "true_positives": overall_metrics["true_positives"],
            "false_positives": overall_metrics["false_positives"],
            "false_negatives": overall_metrics["false_negatives"],
            "true_negatives": overall_metrics["true_negatives"],
        },
        "queries_evaluated": len(query_details),
        "corpus_size": len(corpus),
        "evaluation_time": encoding_time
    }
    
    if progress_callback:
        # Send summary first
        progress_callback({
            "type": "summary_complete",
            "summary": summary_result
        })
        
        # Send each query detail separately
        for i, query_detail in enumerate(query_details):
            progress_callback({
                "type": "query_detail",
                "query_index": i,
                "query_detail": query_detail
            })
            
        # Signal completion
        progress_callback({
            "type": "all_queries_sent",
            "total_queries": len(query_details)
        })
    
    return {
        **summary_result,
        "query_details": query_details  # Keep this for non-streaming usage
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
    parser.add_argument(
        "--max-queries", 
        type=int, 
        help="Maximum number of queries to process (default: all queries)"
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
        max_queries=args.max_queries,
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
