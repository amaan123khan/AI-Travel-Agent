import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import psycopg
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver

from tools import flight_tool, tavily_tool

with patch.object(psycopg, "connect"), patch.object(PostgresSaver, "setup"):
    import main


def _state():
    return {
        "user_query": "Plan a 3-day trip from Delhi to Paris for 2 people, from 10 November to 12 November 2026.",
        "limitations": [],
        "llm_calls": 0,
    }


def _evidence(category="sightseeing", title="Paris visitor guide", url="https://example.test/paris", evidence="Visitor information"):
    return {
        "status": "research",
        "reason": None,
        "items": [{
            "name": title,
            "category": category,
            "source_title": title,
            "source_url": url,
            "retrieved_at": "2026-10-19T00:00:00+00:00",
            "evidence": evidence,
            "verification_status": "research",
            "price": None,
            "currency": None,
            "availability_status": "unavailable",
        }],
    }


class AgentFlowTests(unittest.TestCase):
    def test_tavily_empty_results_and_failure_are_unavailable(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test"}), patch(
            "tavily.TavilyClient.search", return_value={"results": []}
        ):
            empty = tavily_tool.tavily_search("Paris hotel", category="hotel")
        self.assertEqual(empty["status"], "unavailable")
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test"}), patch(
            "tavily.TavilyClient.search", side_effect=RuntimeError("failed")
        ):
            failed = tavily_tool.tavily_search("Paris hotel", category="hotel")
        self.assertEqual(failed["status"], "unavailable")
        self.assertEqual(failed["items"], [])

    def test_tavily_evidence_keeps_complete_source_url_and_research_status(self):
        result_payload = {"results": [{
            "title": "Paris guide",
            "url": "https://example.test/travel/paris",
            "content": "Information from the page.",
        }]}
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test"}), patch(
            "tavily.TavilyClient.search", return_value=result_payload
        ):
            result = tavily_tool.tavily_search("Paris sights", category="sightseeing")
        item = result["items"][0]
        self.assertEqual(item["source_url"], "https://example.test/travel/paris")
        self.assertEqual(item["verification_status"], "research")
        self.assertIsNone(item["price"])
        self.assertIsNone(item["availability_status"])

    def test_itinerary_is_generated_when_flight_search_is_unavailable(self):
        model = Mock()
        model.invoke.return_value = AIMessage(content="Day 1: Sightseeing suggestions.\nDay 2: Explore at your own pace.")
        state = {
            **_state(),
            "flight_results": {"status": "unavailable", "reason": "No response", "items": []},
            "hotel_search_results": {"items": []},
            "restaurant_search_results": {"items": []},
            "sightseeing_search_results": _evidence(),
        }
        with patch.object(main, "_get_llm", return_value=model):
            result = main.itinerary_agent(state)
        self.assertIn("Day 1", result["itinerary"])
        self.assertEqual(model.invoke.call_count, 1)

    def test_unavailable_flights_do_not_create_a_flight_budget_price(self):
        result = main._budget_for_trip(_state()["user_query"])
        flight_cost = next(item for item in result["line_items"] if item["category"] == "International flights")
        self.assertIsNone(flight_cost["amount"])
        self.assertEqual(flight_cost["status"], "unavailable")

    def test_two_night_stay_and_budget_totals_are_correct(self):
        start, end = main._trip_dates(_state()["user_query"])
        self.assertEqual((end - start).days, 2)
        budget = {
            "line_items": [
                {"amount": 1000, "currency": "INR", "status": "verified"},
                {"amount": 2500, "currency": "INR", "status": "estimate"},
                {"amount": None, "currency": "INR", "status": "unavailable"},
            ]
        }
        self.assertEqual(main._budget_totals(budget["line_items"]), {"verified_inr": 1000, "estimate_inr": 2500})
        trip_budget = main._budget_for_trip(_state()["user_query"])
        lodging = next(item for item in trip_budget["line_items"] if item["category"] == "Accommodation")
        self.assertIn("2 hotel nights", lodging["basis"])

    def test_final_keeps_research_as_research_and_links_complete(self):
        result = main.final_agent({
            **_state(),
            "flight_results": {"status": "unavailable", "reason": "No matching schedule", "items": []},
            "hotel_search_results": _evidence("hotel", "Sample hotel", "https://example.test/hotel"),
            "restaurant_search_results": _evidence("restaurant", "Sample restaurant", "https://example.test/food"),
            "sightseeing_search_results": _evidence(),
            "itinerary": "Day 1: Flexible sightseeing suggestion.",
            "budget": main._budget_for_trip(_state()["user_query"]),
        })["final_response"]
        self.assertIn("[Sample hotel](https://example.test/hotel)", result)
        self.assertIn("research evidence", result)
        self.assertIn("2 night(s)", result)
        self.assertIn("What could not be verified", result)
        self.assertIn("No booking has been made", result)
        self.assertNotIn("₹76,000", result)

    def test_frontend_displays_workflow_exception_without_raw_message(self):
        class StopExecution(Exception):
            pass

        fake_streamlit = MagicMock()
        fake_streamlit.status.return_value = Mock()
        fake_streamlit.text_area.return_value = "test query"
        fake_streamlit.button.return_value = True
        fake_streamlit.stop.side_effect = StopExecution()
        with patch.dict(sys.modules, {"streamlit": fake_streamlit}), patch.object(
            main.app, "stream", side_effect=RuntimeError("DATABASE_URL=do-not-show")
        ):
            sys.modules.pop("frontend", None)
            try:
                with self.assertRaises(StopExecution):
                    importlib.import_module("frontend")
            finally:
                sys.modules.pop("frontend", None)
        fake_streamlit.error.assert_called_once_with("Something went wrong (RuntimeError).")


if __name__ == "__main__":
    unittest.main()
