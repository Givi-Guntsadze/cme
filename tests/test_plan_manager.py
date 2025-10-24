import unittest
from datetime import date

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import Activity, CompletedActivity, User, UserPolicy
from app.services.plan import PlanManager, build_plan_with_policy, apply_policy_payloads


class PlanManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        SQLModel.metadata.drop_all(self.engine)

    def _create_user_with_catalog(self) -> User:
        user = User(
            name="Tester",
            target_credits=4.0,
            remaining_credits=4.0,
            budget_usd=500.0,
            allow_live=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        activities = [
            Activity(
                title="Online Safety Course",
                provider="Provider A",
                credits=2.0,
                cost_usd=0.0,
                modality="online",
                start_date=date.today(),
            ),
            Activity(
                title="General Psychiatry Update",
                provider="Provider B",
                credits=2.0,
                cost_usd=100.0,
                modality="online",
            ),
        ]
        self.session.add_all(activities)
        self.session.commit()
        return user

    def test_plan_manager_reuses_active_run_until_stale(self) -> None:
        user = self._create_user_with_catalog()
        manager = PlanManager(self.session)

        run_one = manager.ensure_plan(user, "balanced", policy_bundle={})
        self.assertIsNotNone(run_one.id)
        plan_one, summary_one, _ = manager.serialize_plan(run_one, user)
        self.assertTrue(plan_one)
        self.assertIsNotNone(summary_one)

        run_two = manager.ensure_plan(user, "balanced", policy_bundle={})
        self.assertEqual(run_one.id, run_two.id)

        first_title = plan_one[0]["title"]
        activity_obj = self.session.exec(
            select(Activity).where(Activity.title == first_title)
        ).first()
        self.assertIsNotNone(activity_obj)
        self.session.add(
            CompletedActivity(user_id=user.id, activity_id=activity_obj.id)
        )
        self.session.commit()

        run_three = manager.ensure_plan(user, "balanced", policy_bundle={})
        self.assertNotEqual(run_one.id, run_three.id)
        plan_three, summary_three, _ = manager.serialize_plan(run_three, user)
        self.assertTrue(plan_three)
        self.assertIsNotNone(summary_three)

    def test_build_plan_with_policy_respects_requirement_floor(self) -> None:
        user = User(
            name="Tester",
            target_credits=30.0,
            remaining_credits=30.0,
            budget_usd=5000.0,
            allow_live=True,
            days_off=10,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)

        activities = [
            Activity(
                title="APA Annual Meeting On Demand",
                provider="APA",
                credits=75.0,
                cost_usd=399.0,
                modality="online",
            ),
            Activity(
                title="Live - Key Essentials – Psychiatry CME Event",
                provider="Key Essentials",
                credits=10.0,
                cost_usd=500.0,
                modality="live",
                days_required=3,
            ),
            Activity(
                title="Connecting with Patients for Tobacco Free Living Online Course",
                provider="CCS",
                credits=3.0,
                cost_usd=0.0,
                modality="online",
            ),
            Activity(
                title="Safety Module",
                provider="Safety Org",
                credits=1.0,
                cost_usd=25.0,
                modality="online",
                requirement_tags=["patient_safety"],
            ),
            Activity(
                title="SA-CME Masterclass",
                provider="Masterclass",
                credits=8.0,
                cost_usd=250.0,
                modality="online",
                requirement_tags=["sa_cme"],
            ),
            Activity(
                title="PIP Improvement Project",
                provider="QI Group",
                credits=5.0,
                cost_usd=300.0,
                modality="online",
                requirement_tags=["pip"],
            ),
            Activity(
                title="General Psychiatry Update",
                provider="General Org",
                credits=4.0,
                cost_usd=150.0,
                modality="online",
            ),
            Activity(
                title="Telepsychiatry Series",
                provider="Tele Group",
                credits=3.0,
                cost_usd=120.0,
                modality="online",
            ),
            Activity(
                title="Ethics in Psychiatry",
                provider="Ethics Org",
                credits=2.0,
                cost_usd=90.0,
                modality="online",
            ),
        ]
        self.session.add_all(activities)
        self.session.commit()
        for activity in activities:
            self.session.refresh(activity)

        on_demand = activities[0]
        policy = {
            "remove_titles": [
                "Live - Key Essentials – Psychiatry CME Event",
                "Connecting with Patients for Tobacco Free Living Online Course",
            ]
        }

        recommended, rec_credits, rec_cost, rec_days = build_plan_with_policy(
            user,
            self.session,
            policy,
            mode="balanced",
            remaining_override=0.0,
            budget_override=user.budget_usd,
            days_override=user.days_off,
            exclude_ids={on_demand.id},
        )

        titles = {activity.title for activity in recommended}
        self.assertTrue(titles)
        self.assertGreaterEqual(rec_credits, 16.0)
        self.assertIn("Safety Module", titles)
        self.assertIn("SA-CME Masterclass", titles)
        self.assertIn("PIP Improvement Project", titles)
        self.assertNotIn(
            "Live - Key Essentials – Psychiatry CME Event",
            titles,
        )
        self.assertNotIn(
            "Connecting with Patients for Tobacco Free Living Online Course",
            titles,
        )

    def test_apply_policy_payloads_preserve_remove_titles(self) -> None:
        user = self._create_user_with_catalog()

        apply_policy_payloads(
            ['{"remove_titles": ["Foo Activity", "Bar Activity"]}'],
            user,
            self.session,
            invalidate=False,
            record_message=False,
        )
        self.session.commit()

        apply_policy_payloads(
            ['{"by_mode": {"balanced": {}}}'],
            user,
            self.session,
            invalidate=True,
            record_message=False,
        )
        self.session.commit()

        active_rows = list(
            self.session.exec(
                select(UserPolicy).where(
                    UserPolicy.user_id == user.id, UserPolicy.active.is_(True)
                )
            )
        )
        self.assertTrue(active_rows)
        for row in active_rows:
            payload = row.payload or {}
            remove_titles = payload.get("remove_titles") or []
            self.assertIn("Foo Activity", remove_titles)
            self.assertIn("Bar Activity", remove_titles)


if __name__ == "__main__":
    unittest.main()
