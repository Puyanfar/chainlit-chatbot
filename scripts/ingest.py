"""
Ingestion script for the new FAQ format: a JSON list of {"question", "answer"}
items, with no id (the API assigns ids itself).

Unlike the old version, this script does NOT talk to Qdrant or OpenAI
directly - it sends the data to the standalone FAQ API in batches over HTTP.
That keeps the API as the single place that knows how to correctly embed and
store an FAQ entry; this script is just a bulk client of it.

Safe to re-run: the API derives each entry's id deterministically from its
question text, so re-running with the same question updates that entry in
place rather than creating a duplicate.

Requires the FAQ API to be running first:
    uvicorn api.main:app --host 0.0.0.0 --port 8001

Run:
    python ingest.py
"""

from pathlib import Path
import asyncio
import json
import logging
import sys
import httpx
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = config.EMBEDDING_BATCH_SIZE


def load_faq_data(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        logger.error("FAQ data file not found: %s", file_path)
        sys.exit(1)

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        logger.error("Expected a non-empty JSON list of {question, answer} items.")
        sys.exit(1)

    for item in data:
        if "question" not in item or "answer" not in item:
            logger.error(
                "Each item must have 'question' and 'answer'. Bad item: %s", item
            )
            sys.exit(1)

    return data


async def check_api_available(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get("/health")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def ingest() -> None:
    data = load_faq_data(config.FAQ_DATA_PATH)
    logger.info("Loaded %d FAQ items from %s", len(data), config.FAQ_DATA_PATH)

    created = 0
    updated = 0
    failed = 0

    async with httpx.AsyncClient(
        base_url=config.FAQ_API_BASE_URL, timeout=config.REQUEST_TIMEOUT
    ) as client:

        if not await check_api_available(client):
            logger.error(
                "Can't reach the FAQ API at %s. Is it running? "
                "(uvicorn api.main:app --host 0.0.0.0 --port 8001)",
                config.FAQ_API_BASE_URL,
            )
            sys.exit(1)

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            payload = {
                "items": [
                    {"question": item["question"], "answer": item["answer"]}
                    for item in batch
                ]
            }

            try:
                response = await client.post("/faq/batch", json=payload)
                response.raise_for_status()
            except httpx.HTTPError:
                logger.exception("Batch starting at index %d failed", i)
                failed += len(batch)
                continue

            for result in response.json():
                if result.get("updated"):
                    updated += 1
                else:
                    created += 1

            logger.info(
                "Processed %d/%d items", min(i + BATCH_SIZE, len(data)), len(data)
            )

    logger.info(
        "Done. Created: %d, Updated: %d, Failed: %d, Total: %d",
        created,
        updated,
        failed,
        len(data),
    )


if __name__ == "__main__":
    asyncio.run(ingest())
