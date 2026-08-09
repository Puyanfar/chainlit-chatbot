"""
Standalone FAQ ingestion API.

Run it with:
    uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

Then explore/test it interactively at:
    http://localhost:8001/docs
"""

import logging
import sys
from pathlib import Path

# Let this file import config.py and qdrant_service.py, which live one
# directory up (in the project root, alongside app.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

import config
import qdrant_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the API process starts (not per-request). Makes sure both
    # collections exist before any requests come in.
    await qdrant_service.ensure_collections()
    logger.info(
        "Ready. Collections: '%s' (questions), '%s' (qa)",
        config.QDRANT_QUESTIONS_COLLECTION,
        config.QDRANT_QA_COLLECTION,
    )
    yield


app = FastAPI(title="FAQ Ingestion API", lifespan=lifespan)


# --- Request/response schemas ---
# These Pydantic models define exactly what shape of JSON this API accepts
# and returns. FastAPI uses them to validate incoming requests automatically -
# e.g. a request missing "answer" gets rejected with a clear 422 error before
# any of our code even runs.

class FAQPairIn(BaseModel):
    question: str = Field(..., min_length=1, description="The FAQ question text")
    answer: str = Field(..., min_length=1, description="The FAQ answer text")


class FAQBatchIn(BaseModel):
    items: list[FAQPairIn] = Field(..., min_length=1)


class FAQPairOut(BaseModel):
    id: str
    question: str
    answer: str


# Note: startup logic is handled by the lifespan handler above.


# --- Endpoints ---

@app.get("/health")
async def health():
    """Simple liveness check - useful for confirming the service is up,
    e.g. before wiring monitoring/orchestration to it later."""
    return {"status": "ok"}


@app.post("/faq", response_model=FAQPairOut, status_code=201)
async def add_faq(pair: FAQPairIn):
    """Add a single question/answer pair."""
    try:
        [faq_id] = await qdrant_service.add_faq_pairs([pair.model_dump()])
    except Exception:
        logger.exception("Failed to add FAQ pair")
        raise HTTPException(status_code=500, detail="Failed to add FAQ pair")

    return FAQPairOut(id=faq_id, question=pair.question, answer=pair.answer)


@app.post("/faq/batch", response_model=list[FAQPairOut], status_code=201)
async def add_faq_batch(batch: FAQBatchIn):
    """Add a batch of question/answer pairs in one call."""
    try:
        pairs = [item.model_dump() for item in batch.items]
        faq_ids = await qdrant_service.add_faq_pairs(pairs)
    except Exception:
        logger.exception("Failed to add FAQ batch")
        raise HTTPException(status_code=500, detail="Failed to add FAQ batch")

    return [
        FAQPairOut(id=faq_id, question=item.question, answer=item.answer)
        for faq_id, item in zip(faq_ids, batch.items)
    ]