import os
import re
from datetime import datetime
from typing import TypedDict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search

load_dotenv()
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

class TravelState(TypedDict, total=False):
    user_query: str
    flight_results: dict[str, Any]
    hotel_search_results: dict[str, Any]
    restaurant_search_results: dict[str, Any]
    sightseeing_search_results: dict[str, Any]
    itinerary: str
    budget: str
    limitations: list[str]
    final_response: str
    llm_calls: int

_UNSUPPORTED_TRAVEL_CLAIM = re.compile(
    r"\b(?:flight\s*(?:number|no\.?|#)|airline|boarding pass|ticket price|airfare|"
    r"book(?:ed|ing)?|reservation|confirmed flight|confirmed booking|pnr|gate number)\b",
    re.IGNORECASE,
)

def _traveler_count(query: str) -> int:
    patterns = (
        r"\b(\d{1,2})\s*(?:travelers?|travellers?|people|persons?|adults?)\b",
        r"\bfor\s+(\d{1,2})\s*(?:people|persons?|travelers?|travellers?|adults?)\b",
        r"\b(\d{1,2})\s+of us\b",
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return max(1, min(50, int(match.group(1))))
    return 2

def _filter_unsupported_claims(text: str) -> str:
    """Remove unsupported flight/booking claims without suppressing ordinary itinerary text."""
    if not text:
        return ""
    kept = []
    # Correct splitting: the old pattern containing ** was an invalid regular expression.
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(text)):
        sentence = sentence.strip(" \t-*•")
        if not sentence:
            continue
        clauses = re.split(r"\s*[;|]\s*", sentence)
        safe = [part.strip() for part in clauses if part.strip() and not _UNSUPPORTED_TRAVEL_CLAIM.search(part)]
        if safe:
            kept.append("; ".join(safe))
    return "\n".join(kept).strip()

def _get_llm():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing. Add it to your .env file.")
    return ChatGroq(model=MODEL_NAME, temperature=0.2, api_key=GROQ_API_KEY)

def flight_agent(state: TravelState) -> dict:
    result = search_flights(state.get("user_query", ""))
    limitations = list(state.get("limitations", []))
    if result.get("status") != "verified":
        limitations.append(result.get("reason") or "No verified flight data was returned by AviationStack.")
    return {"flight_results": result, "limitations": limitations}

def hotel_agent(state: TravelState) -> dict:
    query = state.get("user_query", "")
    hotels = tavily_search(f"Hotels and accommodation options for the destination and travel dates in this request: {query}", category="hotel")
    restaurants = tavily_search(f"Restaurants and local food options for the destination in this request: {query}", category="restaurant")
    sights = tavily_search(f"Official visitor information and sightseeing attractions for the destination in this request: {query}", category="sightseeing")
    limitations = list(state.get("limitations", []))
    if hotels.get("status") != "research":
        limitations.append("No useful hotel web results were retrieved; hotel availability and prices are unverified.")
    if restaurants.get("status") != "research":
        limitations.append("No useful restaurant web results were retrieved.")
    if sights.get("status") != "research":
        limitations.append("No useful sightseeing web results were retrieved.")
    return {
        "hotel_search_results": hotels,
        "restaurant_search_results": restaurants,
        "sightseeing_search_results": sights,
        "limitations": limitations,
    }


def _trip_dates(query: str) -> tuple[datetime | None, datetime | None]:
    range_match = re.search(
        r"\bfrom\s+(\d{1,2})\s+([A-Za-z]+)\s+to\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",
        query,
        re.I,
    )
    if range_match:
        try:
            start = datetime.strptime(
                f"{range_match.group(1)} {range_match.group(2)} {range_match.group(5)}", "%d %B %Y"
            )
            end = datetime.strptime(
                f"{range_match.group(3)} {range_match.group(4)} {range_match.group(5)}", "%d %B %Y"
            )
            return start, end
        except ValueError:
            return None, None
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    if len(iso_dates) >= 2:
        try:
            return datetime.strptime(iso_dates[0], "%Y-%m-%d"), datetime.strptime(iso_dates[1], "%Y-%m-%d")
        except ValueError:
            return None, None
    return None, None


def _budget_totals(line_items: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"verified_inr": 0, "estimate_inr": 0}
    for item in line_items:
        amount = item.get("amount")
        if not isinstance(amount, (int, float)) or item.get("currency") != "INR":
            continue
        if item.get("status") == "verified":
            totals["verified_inr"] += int(amount)
        elif item.get("status") == "estimate":
            totals["estimate_inr"] += int(amount)
    return totals


def _budget_for_trip(query: str) -> dict[str, Any]:
    check_in, checkout = _trip_dates(query)
    nights = max(0, (checkout - check_in).days) if check_in and checkout else None
    line_items = [
        {"category": "International flights", "amount": None, "currency": "INR", "status": "unavailable", "basis": "No fare provider is configured."},
        {"category": "Accommodation", "amount": None, "currency": "INR", "status": "unavailable", "basis": f"{nights} hotel nights assumed from check-in/checkout dates; no date-specific rate or availability response." if nights is not None else "Dates or hotel nights could not be determined."},
        {"category": "Food", "amount": None, "currency": "INR", "status": "unavailable", "basis": "No source supplied a defensible trip-specific price."},
        {"category": "Local transport", "amount": None, "currency": "INR", "status": "unavailable", "basis": "No transport fare provider is configured."},
        {"category": "Attractions", "amount": None, "currency": "INR", "status": "unavailable", "basis": "No date-specific ticket prices were supplied."},
        {"category": "Visa", "amount": None, "currency": "INR", "status": "unavailable", "basis": "Traveler eligibility and current fee were not verified."},
        {"category": "Travel insurance", "amount": None, "currency": "INR", "status": "unavailable", "basis": "No policy quote was retrieved."},
    ]
    return {"currency": "INR", "line_items": line_items, "totals": _budget_totals(line_items), "complete_total_available": False}


def _render_evidence(title: str, result: dict[str, Any]) -> str | None:
    items = result.get("items", [])
    if not items:
        return None
    lines = [f"## {title}"]
    for item in items:
        name = item["name"].replace("[", "\\[").replace("]", "\\]")
        url = item["source_url"]
        lines.append(f"- [{name}]({url}) — {item['verification_status']} research evidence; not booking or availability confirmation.")
        lines.append(f"  Evidence: {item['evidence']}")
    return "\n".join(lines)


def _render_flights(result: dict[str, Any]) -> str:
    if result.get("status") != "verified" or not result.get("items"):
        return f"Unavailable: {result.get('reason') or 'No matching provider schedule was returned.'} No fare, booking, or return schedule is available."
    lines = ["Provider-reported outbound schedule fields (not a fare or booking):"]
    for item in result["items"]:
        evidence = item["evidence"]
        departure = evidence.get("departure_scheduled") or evidence.get("departure_estimated") or "time not supplied"
        arrival = evidence.get("arrival_scheduled") or evidence.get("arrival_estimated") or "time not supplied"
        airline = evidence.get("airline") or "Airline not supplied"
        number = evidence.get("flight_number") or "Flight number not supplied"
        lines.append(f"- {airline} {number}: {evidence['departure_iata']} to {evidence['arrival_iata']}; departure {departure}; arrival {arrival}; provider status {evidence.get('provider_status') or 'not supplied'}; verification status: verified.")
        lines.append(f"  Source: [{item['source_title']}]({item['source_url']})")
    lines.append("No fare or seat availability was returned. Return-flight scheduling was not verified.")
    return "\n".join(lines)


def _render_budget(budget: dict[str, Any], travelers: int) -> str:
    lines = ["| Cost category | Amount (INR) | Status | Basis |", "|---|---:|---|---|"]
    for item in budget["line_items"]:
        amount = f"₹{item['amount']:,.0f}" if item["amount"] is not None else "Unavailable"
        treatment = "Included in subtotal" if item["status"] in {"verified", "estimate"} and item["amount"] is not None else "Excluded; amount unavailable"
        lines.append(f"| {item['category']} | {amount} | {item['status']} | {treatment}; {item['basis']} |")
    totals = budget["totals"]
    lines.extend([
        "",
        f"Known verified subtotal: ₹{totals['verified_inr']:,} for {travelers} traveler(s).",
        f"Explicit estimate subtotal: ₹{totals['estimate_inr']:,} for {travelers} traveler(s).",
        "Complete trip total: unavailable because required cost categories have no supported amounts.",
        "No model-generated prices are included.",
    ])
    return "\n".join(lines)

def itinerary_agent(state: TravelState) -> dict:
    llm = _get_llm()
    query = state.get("user_query", "")
    # Flight results are intentionally excluded: missing flight data must not block itinerary generation.
    prompt = f"""Create a useful, realistic travel itinerary based on the request and web evidence.
USER REQUEST:
{query}

HOTEL WEB EVIDENCE:
{state.get('hotel_search_results', {}).get('items', [])}

RESTAURANT WEB EVIDENCE:
{state.get('restaurant_search_results', {}).get('items', [])}

SIGHTSEEING WEB EVIDENCE:
{state.get('sightseeing_search_results', {}).get('items', [])}

Rules:
- Give a day-by-day itinerary with morning, afternoon, and evening suggestions.
- Use specific place names only when supported by evidence or reliably established.
- Never invent flight numbers, airlines, departure/arrival times, ticket fares, bookings, or reservations.
- Do not infer arrival/departure times, full activity days, return travel, hotel availability, or meal reservations from the trip dates.
- Keep activities as flexible suggestions grouped into morning, afternoon, and evening; avoid exact start times and costs.
- Do not claim a hotel or restaurant has availability or that anything is booked.
- Treat all Tavily results as research leads, not specific availability, date-specific prices, or booking confirmations.
- Only cite URLs that appear verbatim in supplied evidence.
- Use the requested destination's known public sights as suggestions, clearly labeled unverified when no source supports them.
- Still create the itinerary if flight information is unavailable.
"""
    response = llm.invoke([
        SystemMessage(content="You are a careful travel planner. Never fabricate live travel data."),
        HumanMessage(content=prompt),
    ])
    itinerary = _filter_unsupported_claims(str(response.content))
    if not itinerary:
        itinerary = "A safe itinerary could not be generated from the model response. Please retry with destination, dates, and interests."

    return {
        "itinerary": itinerary,
        "budget": _budget_for_trip(query),
        "llm_calls": int(state.get("llm_calls", 0)) + 1,
    }

def final_agent(state: TravelState) -> dict:
    limitations = list(dict.fromkeys(state.get("limitations", [])))
    query = state.get("user_query", "")
    start, end = _trip_dates(query)
    overview = "## Trip overview\n"
    if start and end:
        nights = max(0, (end - start).days)
        overview += (
            f"- Dates: {start:%d %B %Y} to {end:%d %B %Y} (inclusive calendar dates).\n"
            f"- Travelers: { _traveler_count(query) }.\n"
            f"- Trip span: {(end - start).days + 1} calendar dates; activity days depend on unverified flight timing.\n"
            f"- Hotel assumption: check-in {start:%d %B}; checkout {end:%d %B}; {nights} night(s).\n"
            "- The date range does not establish actual flight times, arrival times, or full sightseeing days."
        )
    else:
        overview += f"- Travelers: {_traveler_count(query)}. Exact trip/check-in dates could not be determined."
    sections = ["# Your AI Travel Plan", overview, "## Flights and verification status\n" + _render_flights(state.get("flight_results", {}))]
    itinerary = state.get("itinerary", "")
    if itinerary:
        sections.append("## Day-by-day itinerary (suggestions)\n" + itinerary)
    for key, title in (
        ("hotel_search_results", "Hotel research"),
        ("restaurant_search_results", "Restaurant research"),
        ("sightseeing_search_results", "Sightseeing research"),
    ):
        rendered = _render_evidence(title, state.get(key, {}))
        if rendered:
            sections.append(rendered)
    budget = state.get("budget", _budget_for_trip(query))
    sections.append("## Budget breakdown\n" + _render_budget(budget, _traveler_count(query)))
    what_not_verified = limitations + [
        "Specific flight fares, seats, and return schedules.",
        "Date-specific hotel room availability and rates, restaurant reservations, and attraction ticket prices.",
        "Visa eligibility/fees, insurance, food, and local transport costs.",
    ]
    sections.append("## What could not be verified\n" + "\n".join(f"- {item}" for item in dict.fromkeys(what_not_verified)))
    sections.append("**No booking has been made.** Tavily results are research evidence, not booking confirmations.")
    return {"final_response": "\n\n".join(sections)}

def build_app():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing. Add your working PostgreSQL connection string to .env.")
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    graph = StateGraph(TravelState)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_agent", final_agent)
    graph.add_edge(START, "flight_agent")
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_agent")
    graph.add_edge("final_agent", END)
    return graph.compile(checkpointer=checkpointer)

app = build_app()

def generate_trip(user_query: str, thread_id: str = "streamlit-session") -> dict[str, Any]:
    if not user_query or not user_query.strip():
        raise ValueError("Please describe your trip.")
    return app.invoke(
        {"user_query": user_query.strip(), "limitations": [], "llm_calls": 0},
        config={"configurable": {"thread_id": thread_id}},
    )

if __name__ == "__main__":
    print(generate_trip("Plan a 3 day trip from Delhi to Mumbai including hotels and sightseeing.").get("final_response", "No response generated."))
