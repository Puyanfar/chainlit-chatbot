import asyncio
import json, logging, sys, uuid
from pathlib import Path
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

openai_client = AsyncOpenAI(
    base_url=config.API_ENDPOINT,
    api_key=config.API_KEY,
    timeout=config.REQUEST_TIMEOUT,
    max_retries=config.MAX_RETRIES,
)
qdrant_client = AsyncQdrantClient(url=config.QDRANT_URL)


async def get_embedding_dimension():
    result = await openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL, input="dimension check"
    )
    return len(result.data[0].embedding)


def load_faq_data(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        logger.error("FAQ data file not found: %s", file_path)
        sys.exit(1)

    with file_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            logger.error("FAQ data file is not a valid JSON file: %s", e)

    if not isinstance(data, list) or not data:
        logger.error("Expected a non-empty JSON list of {id, context} items.")
        sys.exit(1)

    for item in data:
        if "id" not in item or "context" not in item:
            logger.error("Each item must have 'id' and 'context'. Bad item: %s", item)
            sys.exit(1)

    return data


async def ensure_collection() -> None:
    exists = await qdrant_client.collection_exists(config.QDRANT_COLLECTION_NAME)
    if exists:
        logger.info("Collection '%s' already exists.", config.QDRANT_COLLECTION_NAME)
        return

    EMBEDDING_SIZE = await get_embedding_dimension()

    await qdrant_client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s'.", config.QDRANT_COLLECTION_NAME)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    response = await openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def ingest() -> None:
    data = load_faq_data(config.FAQ_DATA_PATH)
    logger.info("Loaded %d FAQ items from %s", len(data), config.FAQ_DATA_PATH)

    await ensure_collection()

    batch_size = config.EMBEDDING_BATCH_SIZE
    total_upserted = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        texts = [item["context"] for item in batch]

        try:
            vectors = await embed_batch(texts)
        except Exception:
            logger.exception("Embedding failed for batch starting at index %d", i)
            continue

        points = [
            models.PointStruct(
                id=uuid.uuid5(uuid.NAMESPACE_DNS, str(item["id"])),
                vector=vector,
                payload={"context": item["context"]},
            )
            for item, vector in zip(batch, vectors)
        ]

        await qdrant_client.upsert(
            collection_name=config.QDRANT_COLLECTION_NAME,
            points=points,
        )

        total_upserted += len(points)
        logger.info("Upserted %d/%d items", total_upserted, len(data))

    count = await qdrant_client.count(config.QDRANT_COLLECTION_NAME, exact=True)
    logger.info(
        "Done. Collection '%s' now has %d points.",
        config.QDRANT_COLLECTION_NAME,
        count.count,
    )


if __name__ == "__main__":
    asyncio.run(ingest())
