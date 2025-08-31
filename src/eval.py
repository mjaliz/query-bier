import logging
import os
import pathlib
from pathlib import Path

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Just some code to print debug information to stdout
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

corpus, queries, qrels = GenericDataLoader(
    corpus_file=str(corpus_path), query_file=str(query_path), qrels_file=str(qrels_path)
).load_custom()


#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(
    models.SentenceBERT(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        model_kwargs={"default_task": "retrieval.query"},
    ),
    batch_size=16,
)

retriever = EvaluateRetrieval(
    model, score_function="cos_sim"
)  # or "dot" for dot product
results = retriever.retrieve(
    corpus,
    queries,
)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
# util.save_runfile(os.path.join(results_dir, "basalam-jina-v4.run.trec"), results)
util.save_results(
    output_file=os.path.join(results_dir, "basalam-jina-v3.json"),
    ndcg=ndcg,
    _map=_map,
    recall=recall,
    precision=precision,
    mrr=mrr,
)
