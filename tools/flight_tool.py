import os
import re
import logging
from datetime import datetime, timezone
from typing import Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"
NO_VERIFIED_FLIGHT_DATA = "No verified flight data available for the requested route/date."
logger = logging.getLogger(__name__)

CITY_IATA = {
    "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM", "bombay": "BOM",
    "dubai": "DXB", "paris": "CDG", "london": "LHR", "zurich": "ZRH",
    "geneva": "GVA", "singapore": "SIN", "doha": "DOH", "istanbul": "IST",
    "bangkok": "BKK", "kuala lumpur": "KUL", "sydney": "SYD", "melbourne": "MEL",
    "toronto": "YYZ", "vancouver": "YVR", "san francisco": "SFO",
    "los angeles": "LAX", "chicago": "ORD", "new york": "JFK", "tokyo": "HND",
}

def _requested_route(query: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    code_match = re.search(r"\b([A-Z]{3})\s+(?:to|->|→)\s+([A-Z]{3})\b", query, re.I)
    if code_match:
        a, b = code_match.group(1).upper(), code_match.group(2).upper()
        return a, b, a, b
    match = re.search(
        r"\bfrom\s+(?!\d{1,2}\s+[A-Za-z]+\s+to\b)(.+?)\s+(?:to|towards)\s+(.+?)"
        r"(?=\s+(?:on|for|from)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\s+(?:on|for)\s+(?:\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})"
        r"|\s+for\s+\d{1,2}\s*(?:people|persons?|travelers?|travellers?|adults?)\b"
        r"|\s+(?:including|with|and)\b|[,.;]|$)", query, re.I)
    if not match:
        match = re.search(r"\b([A-Za-z][A-Za-z .'-]{1,40}?)\s+to\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?=\s+(?:on|for|from)\b|[,.;]|$)", query, re.I)
    if not match:
        return None, None, None, None
    origin, destination = match.group(1).strip(" ,"), match.group(2).strip(" ,")
    return origin, destination, CITY_IATA.get(origin.lower()), CITY_IATA.get(destination.lower())

def _requested_date(query: str) -> Optional[str]:
    range_match = re.search(
        r"\bfrom\s+(\d{1,2})\s+([A-Za-z]+)\s+to\s+\d{1,2}\s+[A-Za-z]+\s+(\d{4})\b",
        query,
        re.I,
    )
    if range_match:
        try:
            return datetime.strptime(
                f"{range_match.group(1)} {range_match.group(2)} {range_match.group(3)}",
                "%d %B %Y",
            ).date().isoformat()
        except ValueError:
            pass
    formats = [
        (r"\b(\d{4}-\d{2}-\d{2})\b", ["%Y-%m-%d"]),
        (r"\b(\d{1,2}/\d{1,2}/\d{4})\b", ["%m/%d/%Y", "%d/%m/%Y"]),
        (r"\b(\d{1,2}-\d{1,2}-\d{4})\b", ["%d-%m-%Y", "%m-%d-%Y"]),
        (r"\b([A-Za-z]+\s+\d{1,2},?\s+\d{4})\b", ["%B %d %Y", "%b %d %Y"]),
        (r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b", ["%d %B %Y", "%d %b %Y"]),
    ]
    for pattern, candidates in formats:
        match = re.search(pattern, query)
        if not match:
            continue
        value = match.group(1).replace(",", "")
        for fmt in candidates:
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                pass
    return None

def _city_match(provider_text: str, requested_city: str) -> bool:
    if not provider_text or not requested_city:
        return False
    return requested_city.casefold() in provider_text.casefold()

def _unavailable(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "items": [],
        "source": "AviationStack",
        "source_url": AVIATIONSTACK_URL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }


def _redact_secrets(value: str, api_key: str) -> str:
    body = value
    if api_key:
        body = body.replace(api_key, "[REDACTED]")
    body = re.sub(
        r"(?i)((?:access[_-]?(?:key|token)|api[_-]?key|token)[\"'=: ]+)[^&\s,}\"']+",
        r"\1[REDACTED]",
        body,
    )
    return body


def _safe_error_body(response: Any, api_key: str) -> str:
    body = str(getattr(response, "text", "") or "")[:500]
    return _redact_secrets(body, api_key)


def search_flights(query: str) -> dict[str, Any]:
    """Return provider-reported schedules only for an exact route and date."""
    api_key = os.getenv("AVIATIONSTACK_API_KEY", "").strip()
    if not api_key:
        return _unavailable("AVIATIONSTACK_API_KEY is not configured.")
    origin, destination, origin_code, destination_code = _requested_route(query)
    travel_date = _requested_date(query)
    if not origin or not destination:
        return _unavailable("Specify an airport-code or supported city route, such as 'from Delhi to Paris'.")
    if not travel_date:
        return _unavailable("Specify an exact travel date (YYYY-MM-DD).")
    if not origin_code or not destination_code:
        return _unavailable("The requested cities do not map to supported airport codes.")

    params = {"access_key": api_key, "flight_date": travel_date, "limit": 100}
    params["dep_iata"] = origin_code
    params["arr_iata"] = destination_code
    try:
        response = requests.get(AVIATIONSTACK_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return _unavailable("AviationStack request timed out; check the provider or retry later.")
    except requests.HTTPError as exc:
        http_response = exc.response or getattr(exc.request, "response", None)
        status_code = getattr(http_response, "status_code", None)
        body = _safe_error_body(http_response, api_key)
        if status_code in (401, 403):
            reason = "AviationStack rejected authentication or this account's access plan. Check the key and plan permissions."
        elif status_code == 429:
            reason = "AviationStack quota or rate limit was reached. Check account usage and retry later."
        elif status_code == 404:
            reason = "AviationStack endpoint was not found; verify endpoint and plan support."
        else:
            reason = "AviationStack returned an HTTP error."
        logger.warning("AviationStack HTTP failure status=%s body=%s", status_code, body)
        return _unavailable(reason, http_status=status_code, response_excerpt=body)
    except ValueError:
        return _unavailable("AviationStack returned invalid JSON.")
    except requests.RequestException as exc:
        logger.warning("AviationStack transport failure: %s", type(exc).__name__)
        return _unavailable(f"AviationStack transport failed ({type(exc).__name__}).")
    if not isinstance(payload, dict):
        return _unavailable("AviationStack returned a malformed response.")
    if payload.get("error"):
        error = payload["error"]
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        message = _redact_secrets(str(message or "provider returned an error")[:250], api_key)
        logger.warning("AviationStack API error code=%s message=%s", code, message)
        if code in {"invalid_access_key", "missing_access_key", "unauthorized"}:
            reason = "AviationStack rejected the API key; verify AVIATIONSTACK_API_KEY and account access."
        elif code in {"usage_limit_reached", "rate_limit_reached"}:
            reason = "AviationStack account quota or rate limit was reached."
        elif code in {"function_access_restricted", "https_access_restricted"}:
            reason = "AviationStack reports that this endpoint or protocol is not available on the configured plan."
        else:
            reason = f"AviationStack API error: {message}."
        return _unavailable(reason, provider_error_code=code)
    records = payload.get("data")
    if not isinstance(records, list):
        return _unavailable("AviationStack response did not contain a valid data list.")
    if not records:
        return _unavailable(NO_VERIFIED_FLIGHT_DATA)

    matches = []
    for item in records:
        if not isinstance(item, dict):
            continue
        dep = item.get("departure") or {}
        arr = item.get("arrival") or {}
        if not isinstance(dep, dict) or not isinstance(arr, dict):
            continue
        dep_iata, arr_iata = (dep.get("iata") or "").upper(), (arr.get("iata") or "").upper()
        dep_time = dep.get("scheduled") or dep.get("estimated") or ""
        dep_date = dep_time[:10]
        if dep_iata != origin_code or arr_iata != destination_code or dep_date != travel_date:
            continue
        airline = item.get("airline") or {}
        flight = item.get("flight") or {}
        if not isinstance(airline, dict) or not isinstance(flight, dict):
            continue
        matches.append({
            "name": flight.get("iata") or flight.get("number") or airline.get("name") or "Provider schedule",
            "category": "outbound flight schedule",
            "source_title": "AviationStack flight response",
            "source_url": AVIATIONSTACK_URL,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "evidence": {
                "airline": airline.get("name"),
                "flight_number": flight.get("iata") or flight.get("number"),
                "departure_iata": dep_iata,
                "arrival_iata": arr_iata,
                "departure_scheduled": dep.get("scheduled"),
                "departure_estimated": dep.get("estimated"),
                "arrival_scheduled": arr.get("scheduled"),
                "arrival_estimated": arr.get("estimated"),
                "provider_status": item.get("flight_status"),
            },
            "verification_status": "verified",
            "price": None,
            "currency": None,
            "availability_status": None,
        })
    if not matches:
        return _unavailable(NO_VERIFIED_FLIGHT_DATA)
    return {
        "status": "verified",
        "reason": None,
        "items": matches[:8],
        "source": "AviationStack",
        "source_url": AVIATIONSTACK_URL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "requested_route": {"origin": origin_code, "destination": destination_code},
        "requested_date": travel_date,
        "price": None,
        "availability_status": None,
    }
