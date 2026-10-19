import os
import logging
from datetime import datetime, timezone
from urllib.parse import urlsplit
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def tavily_search(
    query: str,
    topic: str = "general",
    max_results: int = 5,
    category: str = "research",
) -> dict:
    """Return timestamped research evidence; search results do not verify bookings."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"status": "unavailable", "reason": "TAVILY_API_KEY is not configured.", "items": [], "retrieved_at": retrieved_at}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            topic=topic,
            max_results=max(1, min(int(max_results), 10)),
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        logger.warning("Tavily search failed: %s", type(exc).__name__)
        return {"status": "unavailable", "reason": f"Tavily search failed ({type(exc).__name__}).", "items": [], "retrieved_at": retrieved_at}
    results = response.get("results", []) if isinstance(response, dict) else []
    if not isinstance(results, list):
        return {"status": "unavailable", "reason": "Tavily returned a malformed results list.", "items": [], "retrieved_at": retrieved_at}
    items = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        content = (result.get("content") or result.get("snippet") or "").strip()
        parsed_url = urlsplit(url)
        if not title or not content or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            continue
        items.append({
            "name": title,
            "category": category,
            "source_title": title,
            "source_url": url,
            "retrieved_at": retrieved_at,
            "evidence": content[:1000],
            "verification_status": "research",
            "price": None,
            "currency": None,
            "availability_status": None,
        })
    if not items:
        return {"status": "unavailable", "reason": "Tavily returned no usable results with complete source URLs.", "items": [], "retrieved_at": retrieved_at}
    return {"status": "research", "reason": None, "items": items, "retrieved_at": retrieved_at}
