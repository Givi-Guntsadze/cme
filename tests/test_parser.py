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


if __name__ == "__main__":
    unittest.main()
