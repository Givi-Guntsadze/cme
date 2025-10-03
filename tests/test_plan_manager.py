import unittest
from datetime import date

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import Activity, CompletedActivity, User
from app.services.plan import PlanManager


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


if __name__ == "__main__":
    unittest.main()
