# AI Multi-Agent Travel Planner

The planner uses LangGraph, Groq, Tavily, PostgreSQL, and Streamlit. Tavily results are retained as timestamped research evidence with source URLs; they do not confirm inventory, date-specific prices, ratings, opening hours, or booking terms. Groq generates a day-wise itinerary independently of flight search. Budget totals include only explicitly priced line items; unsupported costs remain unavailable rather than being model-invented.

The planner does not currently integrate live flight-offer, rail, hotel-inventory, or route-planning APIs. It does not provide verified fares, schedules, booking availability, precise travel times, or a confirmed trip budget.

## Setup

Use Python 3.11 or later and PostgreSQL. From PowerShell in this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add credentials to `.env` (never commit this file):

- `GROQ_API_KEY` — optional attraction ordering with `openai/gpt-oss-120b`. If the ordering request fails, source order is used and the failure is logged.
- `TAVILY_API_KEY` — required for live destination, hotel, restaurant, and sightseeing search.
- `DATABASE_URL` — PostgreSQL connection string used by the LangGraph Postgres checkpointer; required to run the planner.
- `AVIATIONSTACK_API_KEY` — optional AviationStack schedule/status lookup. A request must contain a supported route and explicit outbound date. City names supported by the tool are mapped to airport codes and sent as strict API filters; every returned record must match the route and departure date. HTTP status and a truncated, redacted response excerpt are logged for diagnosis. This integration does not provide bookable offers, seat availability, or fares.

The existing `langgraph_env3` virtual environment can also be used if its dependencies are installed:

```powershell
.\langgraph_env3\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run frontend.py
```

Alternatively, start Streamlit without activating the environment:

```powershell
.\langgraph_env3\Scripts\python.exe -m streamlit run frontend.py
```

The CLI entry point is `python main.py`; it uses the same required PostgreSQL checkpoint store.

## Data limitations

- AviationStack schedules are displayed separately from itinerary suggestions. Missing route/date matches, missing keys, provider errors, and malformed responses are structured as unavailable. A matching provider schedule is not a fare, proof of seat availability, or a booking.
- The trip date span, hotel check-in/checkout assumption, and night count are separate. The date range alone does not establish flight times, full activity days, or return scheduling. A complete total is withheld while required cost categories have no supported prices.
- Tavily evidence includes the retrieved title, complete URL, excerpt, retrieval timestamp, and `research` status. Search results are not independently verified as official, current, available, priced, or bookable.
- Ground transport and rail schedules, fares, and route feasibility are unavailable because no suitable provider is integrated.
- Itinerary suggestions do not include precise timings or geographic optimization. Prices, availability, ratings, fees, and hours are not presented as verified facts.
- Tavily search consumes the configured account's usage quota. No additional travel-data service is configured or silently enabled.

PostgreSQL is still required for the LangGraph checkpointer. The app's startup and live provider behavior depend on the configured credentials and services; successful local checks do not establish live travel availability.
