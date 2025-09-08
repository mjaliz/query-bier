#!/usr/bin/env python3
"""
Script to convert false positives JSONL file to Query format and write to PostgreSQL.
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from config import app_config
from models import Corpus, Query

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_false_positives(file_path: str) -> List[Dict]:
    """Read the false positives JSONL file and return list of records."""
    records = []
    logger.info(f"Reading false positives from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                continue

            if line_num % 50000 == 0:
                logger.info(f"Processed {line_num} lines")

    logger.info(f"Successfully read {len(records)} records")
    return records


def group_by_query_id(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by query_id."""
    grouped = defaultdict(list)

    for record in records:
        query_id = record["query_id"]
        grouped[query_id].append(record)

    logger.info(f"Grouped records into {len(grouped)} unique queries")
    return dict(grouped)


def convert_to_query_format(grouped_records: Dict[str, List[Dict]]) -> List[Query]:
    """Convert grouped records to Query format."""
    queries = []

    for query_id, records in grouped_records.items():
        # Get query text from first record (should be same for all records with same query_id)
        query_text = records[0]["query_text"]

        # Create Corpus objects for each record
        corpuses = []
        for record in records:
            corpus = Corpus(
                corpus_id=record["doc_id"],
                corpus_text=record["doc_text"],
                corpus_query_id=record["doc_q_id"],
                corpus_query_text=record["doc_q_text"],
                qrel_score=record["qrel_score"],
            )
            corpuses.append(corpus.model_dump())

        # Create Query object
        query = Query(query_id=query_id, query_text=query_text, corpuses=corpuses)
        queries.append(query)

        if len(queries) % 100 == 0:
            logger.info(f"Converted {len(queries)} queries")

    logger.info(f"Successfully converted {len(queries)} queries")
    return queries


def create_postgres_connection():
    """Create PostgreSQL connection using config."""
    pg_config = app_config.postgres_config
    connection_string = (
        f"postgresql://{pg_config.USER}:{pg_config.PASSWORD}"
        f"@{pg_config.HOST}:{pg_config.PORT}/{pg_config.DB}"
    )

    engine = create_engine(connection_string, echo=False)
    return engine


def create_tables(engine):
    """Create database tables."""
    logger.info("Creating database tables")
    SQLModel.metadata.create_all(engine)
    logger.info("Tables created successfully")


def write_queries_to_postgres(queries: List[Query], engine):
    """Write Query objects to PostgreSQL."""
    logger.info(f"Writing {len(queries)} queries to PostgreSQL")

    with Session(engine) as session:
        # Add all queries in batches
        batch_size = 1000
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            session.add_all(batch)
            session.commit()
            logger.info(
                f"Written batch {i // batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size}"
            )

    logger.info("Successfully written all queries to PostgreSQL")


def main():
    """Main function to orchestrate the conversion process."""
    try:
        # File path
        file_path = "/Users/mjaliz/basalam/query-bier/data/false_positives_mjaliz-xml-base-gis-basalam-1MQ.jsonl"

        # Step 1: Read false positives file
        logger.info("Step 1: Reading false positives file")
        records = read_false_positives(file_path)

        # Step 2: Group by query_id
        logger.info("Step 2: Grouping records by query_id")
        grouped_records = group_by_query_id(records)

        # Step 3: Convert to Query format
        logger.info("Step 3: Converting to Query format")
        queries = convert_to_query_format(grouped_records)

        # Step 4: Set up PostgreSQL connection
        logger.info("Step 4: Setting up PostgreSQL connection")
        engine = create_postgres_connection()

        # Step 5: Create tables
        logger.info("Step 5: Creating database tables")
        create_tables(engine)

        # Step 6: Write to PostgreSQL
        logger.info("Step 6: Writing queries to PostgreSQL")
        write_queries_to_postgres(queries, engine)

        logger.info("Process completed successfully!")

    except Exception as e:
        logger.error(f"Error during conversion process: {e}")
        raise


if __name__ == "__main__":
    main()
