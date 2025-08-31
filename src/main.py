import asyncio
import json
import math
import time
from datetime import datetime
from itertools import chain
from pathlib import Path

import pandas as pd
from loguru import logger

from src.llm.gemini_client import GeminiClient
from src.llm.openrouter_client import OpenrouterClient
from src.prompt import instruction
from src.rate_limit.token_bucket import AsyncTokenBucket, TokenBucket

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    df = pd.read_csv(DATA_DIR / "long_queries.csv", names=["query", "query_count"])
    queries = df["query"].tolist()
    existing_queries = filter_data()
    quries_set = set(queries)
    eq_set = set(existing_queries)
    queries = list(quries_set - eq_set)
    n = len(keys["api_keys"])
    chunk_size = math.ceil(len(queries) / n)
    chunks = [queries[i : i + chunk_size] for i in range(0, len(queries), chunk_size)]
    data = [
        {"api_key": keys["api_keys"][i], "queries": chunks[i][:50]}
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


async def run_openrouter():
    while True:
        data = load_all_data()[:5]
        if not data:
            logger.info("completed")
            return
        client = OpenrouterClient(
            api_key="",
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
