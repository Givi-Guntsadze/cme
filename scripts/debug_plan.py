"""Debug script to check what plan data exists in the database."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_session
from sqlmodel import select
from app.models import PlanRun, PlanItem, User, Activity

def main():
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            print("No user found!")
            return
        
        print(f"User ID: {user.id}, Name: {user.name}")
        print(f"Target Credits: {user.target_credits}, Remaining: {user.remaining_credits}")
        print()
        
        # Get all plan runs
        runs = list(session.exec(select(PlanRun).where(PlanRun.user_id == user.id)))
        print(f"Total Plan Runs: {len(runs)}")
        
        for run in runs[:5]:
            items = list(session.exec(select(PlanItem).where(PlanItem.plan_run_id == run.id)))
            print(f"  Run ID={run.id}, Mode={run.mode}, Items={len(items)}")
            
            for item in items[:3]:
                activity = session.get(Activity, item.activity_id) if item.activity_id else None
                title = activity.title if activity else "(No activity)"
                print(f"    - {item.position}. {title} (committed={item.committed})")
            
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more items")
        
        # Check what ensure_plan would return
        print()
        print("=== Testing ensure_plan behavior ===")
        from app.services.plan import PlanManager
        pm = PlanManager(session)
        
        # This is what _state_snapshot calls
        test_run = pm.ensure_plan(user, "balanced", policy_bundle=None)
        test_items = list(session.exec(select(PlanItem).where(PlanItem.plan_run_id == test_run.id)))
        print(f"ensure_plan(policy_bundle=None) returned: Run ID={test_run.id}, Items={len(test_items)}")
        
        # Check latest run by ID
        latest_run = max(runs, key=lambda r: r.id) if runs else None
        if latest_run:
            latest_items = list(session.exec(select(PlanItem).where(PlanItem.plan_run_id == latest_run.id)))
            print(f"Latest run by ID: Run ID={latest_run.id}, Items={len(latest_items)}")

if __name__ == "__main__":
    main()
