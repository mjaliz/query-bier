import asyncio
import json
import math
import time
from datetime import datetime
from itertools import chain
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from sqlmodel import Session, select

from src.config import app_config
from src.llm.gemini_client import GeminiClient
from src.llm.openrouter_client import OpenrouterClient
from src.models import BeirScore, Query
from src.prompt import instruction
from src.rate_limit.token_bucket import AsyncTokenBucket

DATA_DIR = Path(__file__).parent.parent / "data"


def create_postgres_connection():
    """Create PostgreSQL connection using config."""
    pg_config = app_config.postgres_config
    connection_string = (
        f"postgresql://{pg_config.USER}:{pg_config.PASSWORD}"
        f"@{pg_config.HOST}:{pg_config.PORT}/{pg_config.DB}"
    )
    engine = create_engine(connection_string, echo=False)
    return engine


def load_data():
    df = pd.read_csv(DATA_DIR / "long_queries.csv", names=["query", "query_count"])
    queries = df["query"].tolist()
    existing_queries = filter_data()
    quries_set = set(queries)
    eq_set = set(existing_queries)
    queries = list(quries_set - eq_set)
    api_keys = app_config.api_config.GEMINI_API_KEYS
    if not api_keys:
        logger.warning("No API keys configured")
        return []
    n = len(api_keys)
    chunk_size = math.ceil(len(queries) / n)
    chunks = [queries[i : i + chunk_size] for i in range(0, len(queries), chunk_size)]
    data = [
        {"api_key": api_keys[i], "queries": chunks[i][:50]}
        for i in range(0, len(chunks))
    ]
    return data


def load_all_data():
    df = pd.read_csv(DATA_DIR / "long_queries.csv", names=["query", "query_count"])
    queries = df["query"].tolist()
    existing_queries = filter_data()
    quries_set = set(queries)
    eq_set = set(existing_queries)
    queries = list(quries_set - eq_set)
    return queries


def filter_data():
    paths = (DATA_DIR / "..").glob("*.json")
    queries = []
    for p in paths:
        with open(p, "r") as f:
            data = json.loads(f.read())
            queries.extend([i["query"] for i in data])
    return queries


async def get_bier(query, client: OpenrouterClient):
    logger.info("getting res for {}", query)
    prompt = instruction.format(query=query)
    res = await client.generate_text(prompt=prompt)
    if not res:
        logger.error("Failed to generate text")
        return None
    return res, query


async def get_bier_score(query, corpus_list, client: OpenrouterClient):
    logger.info("getting res for {}", query)
    prompt = instruction.format(query=query, corpus_list=corpus_list)
    res = await client.generate_text(prompt=prompt, response_format=BeirScore)
    if not res:
        logger.error("Failed to generate text")
        return None
    return res, query


async def run_bier_score():
    """
    Main function to run BEIR scoring process:
    1. Get queries without results from database
    2. Run get_bier_score for each query
    3. Update the results in the database
    """
    try:
        # Create database connection
        engine = create_postgres_connection()

        with Session(engine) as session:
            # Get queries that don't have results yet
            statement = select(Query).where(Query.results.is_(None))
            queries_without_results = session.exec(statement).all()

            if not queries_without_results:
                logger.info("No queries without results found")
                return

            logger.info(f"Found {len(queries_without_results)} queries without results")

            # Initialize OpenRouter client
            api_key = app_config.api_config.OPENROUTER_API_KEY
            if not api_key:
                logger.error("OpenRouter API key not configured")
                return

            client = OpenrouterClient(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                limiter=AsyncTokenBucket(capacity=200, rate=1000 / 60),
            )

            # Process queries in batches to avoid overwhelming the API
            batch_size = 10
            for i in range(0, len(queries_without_results), batch_size):
                batch = queries_without_results[i : i + batch_size]

                logger.info(
                    f"Processing batch {i // batch_size + 1}/{(len(queries_without_results) + batch_size - 1) // batch_size}"
                )

                # Prepare tasks for concurrent processing
                tasks = []
                for query in batch:
                    # Convert corpus list to the format expected by the prompt
                    corpus_list = [
                        {
                            "corpus_id": corpus.corpus_id,
                            "corpus_text": corpus.corpus_text,
                        }
                        for corpus in query.corpuses
                    ]

                    task = get_bier_score(
                        query=query.query_text, corpus_list=corpus_list, client=client
                    )
                    tasks.append((query, task))

                # Execute all tasks concurrently
                results = await asyncio.gather(
                    *[task for _, task in tasks], return_exceptions=True
                )

                # Process results and update database
                for (query, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error processing query {query.query_id}: {result}"
                        )
                        continue

                    if result is None:
                        logger.warning(f"No result returned for query {query.query_id}")
                        continue

                    try:
                        # Parse the result and convert to BeirScore objects
                        llm_response, _ = result

                        # Assuming the LLM response contains a list of scored items
                        # The exact structure depends on your LLM client implementation
                        if hasattr(llm_response, "curpos"):
                            # Convert curpos to BeirScore format
                            beir_scores = []
                            for item in llm_response.curpos:
                                beir_score = BeirScore(
                                    corpus_id=item.corpus_id,
                                    corpus_text=item.corpus_text,
                                    score=item.score,
                                )
                                beir_scores.append(beir_score)

                            # Update the query with results
                            query.results = beir_scores
                            session.add(query)

                            logger.info(
                                f"Updated query {query.query_id} with {len(beir_scores)} scored results"
                            )
                        else:
                            logger.warning(
                                f"Unexpected response format for query {query.query_id}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error processing result for query {query.query_id}: {e}"
                        )
                        continue

                # Commit the batch
                try:
                    session.commit()
                    logger.info(f"Successfully committed batch {i // batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error committing batch: {e}")
                    session.rollback()

                # Add a small delay between batches to respect rate limits
                await asyncio.sleep(1)

        logger.info("BEIR scoring process completed successfully")

    except Exception as e:
        logger.error(f"Error in run_bier_score: {e}")
        raise


async def run_openrouter():
    while True:
        data = load_all_data()[:5]
        if not data:
            logger.info("completed")
            return
        api_key = app_config.api_config.OPENROUTER_API_KEY
        if not api_key:
            logger.error("OpenRouter API key not configured")
            return

        client = OpenrouterClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            limiter=AsyncTokenBucket(capacity=200, rate=1000 / 60),
        )
        tasks = [get_bier(query=query, client=client) for query in data]
        res = await asyncio.gather(*tasks)
        results = []
        for r in res:
            if r is None:
                logger.info("result is None")
                continue
            results.append(
                {"query": r[1], "curpos": [cr.model_dump() for cr in r[0].curpos]}
            )

        with open(f"res_{datetime.now()}.json", "w") as f:
            f.write(json.dumps(results, ensure_ascii=False))
        time.sleep(60)


async def run():
    while True:
        data = load_data()
        if not data:
            logger.info("completed")
            return
        clients = [
            GeminiClient(
                data[i]["api_key"], AsyncTokenBucket(capacity=10, rate=10 / 60)
            )
            for i in range(0, len(data))
        ]
        tasks = [
            [get_bier(query, clients[i]) for query in data[i]["queries"]]
            for i in range(0, len(data))
        ]
        res = await asyncio.gather(*list(chain.from_iterable(tasks)))
        results = []
        for r in res:
            if r is None:
                logger.info("result is None")
                continue
            results.append(
                {"query": r[1], "curpos": [cr.model_dump() for cr in r[0].curpos]}
            )

        with open(f"res_{datetime.now()}.json", "w") as f:
            f.write(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(run_openrouter())
