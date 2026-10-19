import unittest
from unittest.mock import Mock, patch

from tools import flight_tool


def _flight(origin="DEL", destination="CDG", date="2026-11-10"):
    return {
        "airline": {"name": "Example Air"},
        "flight": {"iata": "EA100"},
        "flight_status": "scheduled",
        "departure": {"iata": origin, "scheduled": f"{date}T08:00:00+00:00"},
        "arrival": {"iata": destination, "scheduled": f"{date}T18:00:00+00:00"},
    }


class SearchFlightsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"AVIATIONSTACK_API_KEY": "test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_missing_key_is_structured_unavailable(self):
        with patch.dict("os.environ", {}, clear=True):
            result = flight_tool.search_flights("Delhi to Paris on 2026-11-10")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("AVIATIONSTACK_API_KEY", result["reason"])

    @patch("tools.flight_tool.requests.get")
    def test_http_error_records_status_and_redacted_body(self, get):
        response = Mock(status_code=403, text='{"error":"bad key test-secret"}')
        get.side_effect = flight_tool.requests.HTTPError(response=response)
        result = flight_tool.search_flights("Delhi to Paris on 2026-11-10")
        self.assertEqual(result["http_status"], 403)
        self.assertNotIn("test-secret", result["response_excerpt"])
        self.assertIn("rejected authentication", result["reason"])

    @patch("tools.flight_tool.requests.get")
    def test_timeout_is_reported_separately(self, get):
        get.side_effect = flight_tool.requests.Timeout()
        result = flight_tool.search_flights("Delhi to Paris on 2026-11-10")
        self.assertIn("timed out", result["reason"])

    @patch("tools.flight_tool.requests.get")
    def test_provider_auth_error_is_classified_and_redacted(self, get):
        response = Mock()
        response.json.return_value = {
            "error": {"code": "invalid_access_key", "message": "invalid token test-secret"}
        }
        get.return_value = response
        result = flight_tool.search_flights("Delhi to Paris on 2026-11-10")
        self.assertEqual(result["provider_error_code"], "invalid_access_key")
        self.assertNotIn("test-secret", result["reason"])
        self.assertIn("rejected the API key", result["reason"])

    @patch("tools.flight_tool.requests.get")
    def test_empty_flight_response_is_unavailable(self, get):
        response = Mock()
        response.json.return_value = {"data": []}
        get.return_value = response
        result = flight_tool.search_flights("Delhi to Paris on 2026-11-10")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["items"], [])

    @patch("tools.flight_tool.requests.get")
    def test_rejects_wrong_route_and_date(self, get):
        response = Mock()
        response.json.return_value = {"data": [
            _flight("DEL", "CDG", "2026-11-11"),
            _flight("CDG", "DEL", "2026-11-10"),
            _flight("DEL", "CDG", ""),
        ]}
        get.return_value = response
        result = flight_tool.search_flights("Delhi to Paris on 2026-11-10")
        self.assertEqual(result["status"], "unavailable")

    @patch("tools.flight_tool.requests.get")
    def test_valid_match_is_structured_and_has_no_fare_claim(self, get):
        response = Mock()
        response.json.return_value = {"data": [_flight()]}
        get.return_value = response
        result = flight_tool.search_flights("Delhi to Paris on 2026-11-10")
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["items"][0]["verification_status"], "verified")
        self.assertIsNone(result["items"][0]["price"])
        self.assertIsNone(result["items"][0]["availability_status"])
        self.assertEqual(get.call_args.kwargs["params"]["dep_iata"], "DEL")
        self.assertEqual(get.call_args.kwargs["params"]["arr_iata"], "CDG")

    @patch("tools.flight_tool.requests.get")
    def test_range_uses_first_date_and_strict_codes(self, get):
        response = Mock()
        response.json.return_value = {"data": [_flight()]}
        get.return_value = response
        result = flight_tool.search_flights(
            "Plan a trip from Delhi to Paris for 2 people, from 10 November to 12 November 2026"
        )
        self.assertEqual(result["requested_date"], "2026-11-10")
        self.assertEqual(get.call_args.kwargs["params"]["flight_date"], "2026-11-10")


if __name__ == "__main__":
    unittest.main()
