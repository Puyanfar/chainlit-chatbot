"""
Calibration helper: run a handful of test queries against your Qdrant collection
and print the top-k scores for each, so you can pick a sensible RAG_SCORE_THRESHOLD.

Include a mix of:
  - queries you KNOW should match something in the FAQ
  - queries you KNOW are unrelated (small talk, off-topic questions)

Look at the gap between the two groups' scores to choose a threshold that sits
between them.

Run:
    python scripts/check_scores.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.clients import embedding_client, qdrant_client, embed_texts
import config


# Edit this list with real examples from your own domain.
TEST_QUERIES = [
    "ایمیل منابع انسانی سبحان چیه؟",       # expect: relevant, if this is in your FAQ
    "توی سبحان از چه زبان‌های برنامه نویسی استفاده میشه؟",      # expect: relevant, if this is in your FAQ
    "تفاوت جاوا و پایتون چیه؟",     # expect: irrelevant
    "خودت رو معرفی کن.",                     # expect: irrelevant
]


async def main():

    for query in TEST_QUERIES:
        [vector] = await embed_texts([query])

        results_qa = await qdrant_client.query_points(
            collection_name=config.QDRANT_QA_COLLECTION,
            query=vector,
            limit=3,
        )

        results_questions = await qdrant_client.query_points(
            collection_name=config.QDRANT_QUESTIONS_COLLECTION,
            query=vector,
            limit=3,
        )

        print(f"\nQuery: {query!r}")
        if not results_qa.points and not results_questions.points:
            print("  (collection empty or no results)")
            continue

        for point in results_qa.points:
            snippet = (point.payload or {}).get("context", "")[:80]
            print(f"  score={point.score:.4f}  {snippet!r}")

        for point in results_questions.points:
            snippet = (point.payload or {}).get("context", "")[:80]
            print(f"  score={point.score:.4f}  {snippet!r}")


if __name__ == "__main__":
    asyncio.run(main())