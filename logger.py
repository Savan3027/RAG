# logger.py

import logging

logging.basicConfig(
    filename="rag.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def log_query(query, retrieved_chunks, answer):
    logging.info(f"Query: {query}")
    logging.info(f"Retrieved: {retrieved_chunks}")
    logging.info(f"Answer: {answer}")