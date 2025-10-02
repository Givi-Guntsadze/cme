import unittest
from datetime import date
from unittest.mock import patch

from sqlmodel import SQLModel, create_engine, Session

from app.models import User, Activity, Claim
from app.planner import build_plan
from app.requirements import validate_against_requirements


class RequirementsUsageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        SQLModel.metadata.drop_all(self.engine)

    def test_validate_against_requirements_uses_cycle_total_and_patient_safety_flag(
        self,
    ) -> None:
        user = User(name="Tester", target_credits=0, remaining_credits=0)
        self.session.add(user)
        self.session.commit()

        req_data = {
            "board": "ABPN",
            "specialty": "Psychiatry",
            "rules": {
                "cme_total_per_cycle": 90,
                "patient_safety_activity": {
                    "required": True,
                    "portal_determined": True,
                },
            },
        }
        result = validate_against_requirements(user, [], req_data)

        self.assertEqual(result["summary"]["target_total"], 90)
        checks = {item["label"]: item for item in result["checks"]}
        self.assertIn("Patient safety activity", checks)
        self.assertEqual(checks["Patient safety activity"]["status"], "warn")
        self.assertIn("Portal", checks["Patient safety activity"]["detail"])

    def test_validate_includes_sa_cme_and_pip(self) -> None:
        user = User(name="Tester", target_credits=0, remaining_credits=0)
        self.session.add(user)
        self.session.commit()

        claims = [
            Claim(
                user_id=user.id,
                credits=2.0,
                topic="Self-Assessment CME",
                date=date.today(),
                source_text="Completed SA-CME knowledge self-assessment",
            ),
            Claim(
                user_id=user.id,
                credits=0.0,
                topic="Performance Improvement Project",
                date=date.today(),
                source_text="Finished PIP chart review and reassessment",
            ),
        ]
        self.session.add_all(claims)
        self.session.commit()

        req_data = {
            "board": "ABPN",
            "specialty": "Psychiatry",
            "rules": {
                "total_credits": 40,
                "sa_cme_min_per_cycle": 6,
                "pip_required_per_cycle": 1,
            },
        }
        result = validate_against_requirements(user, claims, req_data)
        checks = {item["label"]: item for item in result["checks"]}

        self.assertEqual(checks["Self-Assessment CME"]["status"], "fail")
        self.assertIn("2.0", checks["Self-Assessment CME"]["detail"])
        self.assertEqual(checks["Performance Improvement (PIP)"]["status"], "ok")

    @patch("app.planner.load_abpn_psychiatry_requirements")
    def test_plan_generation_prioritizes_patient_safety_when_required(
        self, mock_load
    ) -> None:
        mock_load.return_value = {
            "rules": {
                "total_credits": 2,
                "patient_safety_activity": {"required": True},
            }
        }
        user = User(
            name="Tester",
            target_credits=2,
            remaining_credits=2,
            budget_usd=500,
            allow_live=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        general = Activity(
            title="General CME Update",
            provider="Provider A",
            credits=2.0,
            cost_usd=0.0,
            modality="online",
        )
        safety = Activity(
            title="Patient Safety Workshop",
            provider="Provider B",
            credits=2.0,
            cost_usd=150.0,
            modality="online",
        )
        self.session.add_all([general, safety])
        self.session.commit()

        chosen, total_credits, total_cost, _ = build_plan(
            user, self.session, mode="cheapest"
        )

        chosen_titles = {activity.title for activity in chosen}
        self.assertIn("Patient Safety Workshop", chosen_titles)
        self.assertGreaterEqual(total_credits, user.remaining_credits)
        self.assertEqual(total_cost, 150.0)
        tags_sets = [
            set(getattr(activity, "_requirement_tags", set()) or [])
            for activity in chosen
        ]
        self.assertTrue(any("patient_safety" in tags for tags in tags_sets))

    @patch("app.planner.load_abpn_psychiatry_requirements")
    def test_plan_reverts_to_cost_when_patient_safety_already_met(
        self, mock_load
    ) -> None:
        mock_load.return_value = {
            "rules": {
                "total_credits": 2,
                "patient_safety_activity": {"required": True},
            }
        }
        user = User(
            name="Tester",
            target_credits=2,
            remaining_credits=2,
            budget_usd=500,
            allow_live=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        claim = Claim(
            user_id=user.id,
            credits=1.0,
            topic="Patient safety and quality",
            date=date.today(),
            source_text="Completed patient safety module",
        )
        self.session.add(claim)

        general = Activity(
            title="General CME Update",
            provider="Provider A",
            credits=2.0,
            cost_usd=0.0,
            modality="online",
        )
        safety = Activity(
            title="Patient Safety Workshop",
            provider="Provider B",
            credits=2.0,
            cost_usd=150.0,
            modality="online",
        )
        self.session.add_all([general, safety])
        self.session.commit()

        chosen, total_credits, total_cost, _ = build_plan(
            user, self.session, mode="cheapest"
        )
        chosen_titles = {activity.title for activity in chosen}
        self.assertIn("General CME Update", chosen_titles)
        self.assertNotIn("Patient Safety Workshop", chosen_titles)
        self.assertEqual(total_cost, 0.0)
        tags_sets = [
            set(getattr(activity, "_requirement_tags", set()) or [])
            for activity in chosen
        ]
        self.assertTrue(all("patient_safety" not in tags for tags in tags_sets))

    @patch("app.planner.load_abpn_psychiatry_requirements")
    def test_plan_prioritizes_sa_cme_when_short(self, mock_load) -> None:
        mock_load.return_value = {
            "rules": {
                "total_credits": 4,
                "sa_cme_min_per_cycle": 4,
            }
        }
        user = User(
            name="Tester",
            target_credits=4,
            remaining_credits=4,
            budget_usd=500,
            allow_live=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        sa_activity = Activity(
            title="Self-Assessment Knowledge Review",
            provider="Provider B",
            credits=2.0,
            cost_usd=250.0,
            modality="online",
        )
        general = Activity(
            title="General CME Update",
            provider="Provider A",
            credits=2.0,
            cost_usd=0.0,
            modality="online",
        )
        general_alt = Activity(
            title="General CME Deep Dive",
            provider="Provider C",
            credits=2.0,
            cost_usd=0.0,
            modality="online",
        )
        self.session.add_all([sa_activity, general, general_alt])
        self.session.commit()

        chosen, total_credits, total_cost, _ = build_plan(
            user, self.session, mode="cheapest"
        )
        chosen_titles = {activity.title for activity in chosen}
        self.assertIn("Self-Assessment Knowledge Review", chosen_titles)
        self.assertGreaterEqual(total_cost, 250.0)
        tags_sets = [
            set(getattr(activity, "_requirement_tags", set()) or [])
            for activity in chosen
        ]
        self.assertTrue(any("sa_cme" in tags for tags in tags_sets))

    @patch("app.planner.load_abpn_psychiatry_requirements")
    def test_plan_skips_sa_cme_when_requirement_fulfilled(self, mock_load) -> None:
        mock_load.return_value = {
            "rules": {
                "total_credits": 4,
                "sa_cme_min_per_cycle": 4,
            }
        }
        user = User(
            name="Tester",
            target_credits=4,
            remaining_credits=4,
            budget_usd=500,
            allow_live=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        claim = Claim(
            user_id=user.id,
            credits=4.0,
            topic="Self-Assessment CME",
            date=date.today(),
            source_text="Completed SA-CME knowledge self-assessment",
        )
        self.session.add(claim)

        sa_activity = Activity(
            title="Self-Assessment Knowledge Review",
            provider="Provider B",
            credits=2.0,
            cost_usd=250.0,
            modality="online",
        )
        general = Activity(
            title="General CME Update",
            provider="Provider A",
            credits=2.0,
            cost_usd=0.0,
            modality="online",
        )
        general_alt = Activity(
            title="General CME Deep Dive",
            provider="Provider C",
            credits=2.0,
            cost_usd=0.0,
            modality="online",
        )
        self.session.add_all([sa_activity, general, general_alt])
        self.session.commit()

        chosen, _, total_cost, _ = build_plan(user, self.session, mode="cheapest")
        chosen_titles = {activity.title for activity in chosen}
        self.assertNotIn("Self-Assessment Knowledge Review", chosen_titles)
        self.assertEqual(total_cost, 0.0)
        tags_sets = [
            set(getattr(activity, "_requirement_tags", set()) or [])
            for activity in chosen
        ]
        self.assertTrue(all("sa_cme" not in tags for tags in tags_sets))

    @patch("app.planner.load_abpn_psychiatry_requirements")
    def test_plan_prioritizes_pip_activity_when_required(self, mock_load) -> None:
        mock_load.return_value = {
            "rules": {
                "total_credits": 2,
                "pip_required_per_cycle": 1,
            }
        }
        user = User(
            name="Tester",
            target_credits=2,
            remaining_credits=2,
            budget_usd=500,
            allow_live=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        pip_activity = Activity(
            title="Performance Improvement Project Workshop",
            provider="Provider P",
            credits=1.0,
            cost_usd=300.0,
            modality="online",
        )
        general = Activity(
            title="General CME Update",
            provider="Provider A",
            credits=1.0,
            cost_usd=0.0,
            modality="online",
        )
        extra = Activity(
            title="Another CME Update",
            provider="Provider C",
            credits=1.0,
            cost_usd=0.0,
            modality="online",
        )
        self.session.add_all([pip_activity, general, extra])
        self.session.commit()

        chosen, _, total_cost, _ = build_plan(user, self.session, mode="cheapest")
        chosen_titles = {activity.title for activity in chosen}
        self.assertIn("Performance Improvement Project Workshop", chosen_titles)
        self.assertGreaterEqual(total_cost, 300.0)
        tags_sets = [
            set(getattr(activity, "_requirement_tags", set()) or [])
            for activity in chosen
        ]
        self.assertTrue(any("pip" in tags for tags in tags_sets))

    @patch("app.planner.load_abpn_psychiatry_requirements")
    def test_plan_skips_pip_when_requirement_satisfied(self, mock_load) -> None:
        mock_load.return_value = {
            "rules": {
                "total_credits": 2,
                "pip_required_per_cycle": 1,
            }
        }
        user = User(
            name="Tester",
            target_credits=2,
            remaining_credits=2,
            budget_usd=500,
            allow_live=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        claim = Claim(
            user_id=user.id,
            credits=1.0,
            topic="Performance Improvement project",
            date=date.today(),
            source_text="Completed PIP module",
        )
        self.session.add(claim)

        pip_activity = Activity(
            title="Performance Improvement Project Workshop",
            provider="Provider P",
            credits=1.0,
            cost_usd=300.0,
            modality="online",
        )
        general = Activity(
            title="General CME Update",
            provider="Provider A",
            credits=1.0,
            cost_usd=0.0,
            modality="online",
        )
        general_alt = Activity(
            title="General CME Refresher",
            provider="Provider C",
            credits=1.0,
            cost_usd=0.0,
            modality="online",
        )
        self.session.add_all([pip_activity, general, general_alt])
        self.session.commit()

        chosen, _, total_cost, _ = build_plan(user, self.session, mode="cheapest")
        chosen_titles = {activity.title for activity in chosen}
        self.assertIn("General CME Update", chosen_titles)
        self.assertNotIn("Performance Improvement Project Workshop", chosen_titles)
        self.assertEqual(total_cost, 0.0)
        tags_sets = [
            set(getattr(activity, "_requirement_tags", set()) or [])
            for activity in chosen
        ]
        self.assertTrue(all("pip" not in tags for tags in tags_sets))


if __name__ == "__main__":
    unittest.main()
