import unittest
from datetime import date

from app.parser import parse_message


class ParserTests(unittest.TestCase):
    def test_ignores_year_in_title(self) -> None:
        credits, topic, parsed_date = parse_message(
            "I'm done with 2023 CME Activities Guide"
        )
        self.assertEqual(credits, 0.0)
        self.assertEqual(topic, None)
        self.assertIsInstance(parsed_date, date)

    def test_detects_explicit_credits(self) -> None:
        credits, topic, parsed_date = parse_message("Logged 2.5 credits for safety")
        self.assertAlmostEqual(credits, 2.5)
        self.assertEqual(topic, "safety")
        self.assertIsInstance(parsed_date, date)

    def test_preference_statement_skips_claim(self) -> None:
        credits, topic, parsed_date = parse_message(
            "I'd rather attend one conference and get 45 credits"
        )
        self.assertEqual(credits, 0.0)
        self.assertEqual(topic, None)
        self.assertIsInstance(parsed_date, date)

    def test_delete_request_skips_claim(self) -> None:
        credits, topic, parsed_date = parse_message(
            "Please delete the mistaken 112.5 credits entry from my log"
        )
        self.assertEqual(credits, 0.0)
        self.assertIsInstance(parsed_date, date)

    def test_negated_completion_skips_claim(self) -> None:
        credits, topic, parsed_date = parse_message(
            "I didn't actually complete those 5 credits yet"
        )
        self.assertEqual(credits, 0.0)
        self.assertIsInstance(parsed_date, date)


class ActivityUpdateTests(unittest.TestCase):
    """Tests for _extract_activity_update function."""
    
    def setUp(self):
        from app.main import _extract_activity_update
        self.extract = _extract_activity_update
    
    def test_cost_update_simple(self):
        result = self.extract("Opioid CME costs $125")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("cost"), 125.0)
        self.assertIn("opioid", result.get("title", "").lower())
    
    def test_cost_update_with_eligibility(self):
        result = self.extract("The Opioid Training costs $50 and I am eligible")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("cost"), 50.0)
        self.assertEqual(result.get("eligible"), True)
    
    def test_eligibility_only(self):
        result = self.extract("I'm eligible for the Patient Safety Course")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("eligible"), True)
        self.assertIn("patient safety", result.get("title", "").lower())
    
    def test_no_match_returns_none(self):
        result = self.extract("What's the weather like?")
        # Should return None or empty result
        self.assertTrue(result is None or result.get("title") is None)


class DiscoveryQueryTests(unittest.TestCase):
    """Tests for _extract_discovery_query function."""
    
    def setUp(self):
        from app.main import _extract_discovery_query
        self.extract = _extract_discovery_query
    
    def test_find_me_pattern(self):
        result = self.extract("find me cheap opioid courses")
        self.assertIsNotNone(result)
        self.assertIn("opioid", result.lower())
    
    def test_search_for_pattern(self):
        result = self.extract("search for patient safety activities")
        self.assertIsNotNone(result)
        self.assertIn("patient safety", result.lower())
    
    def test_show_me_pattern(self):
        result = self.extract("show me leadership CME")
        self.assertIsNotNone(result)
        self.assertIn("leadership", result.lower())
    
    def test_any_courses_pattern(self):
        result = self.extract("any courses on addiction medicine")
        self.assertIsNotNone(result)
        self.assertIn("addiction", result.lower())
    
    def test_non_discovery_returns_none(self):
        result = self.extract("I completed 5 credits today")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
