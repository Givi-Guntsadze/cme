#!/usr/bin/env python
"""Debug script to test removal tool."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import select
from app.db import get_session
from app.models import User, Activity, PlanItem, PlanRun
from app.services.tools import remove_activity
from app.services.discovery import find_activity_by_title

def test_removal():
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            print("No user found!")
            return
        
        print(f"User: {user.name} (ID: {user.id})")
        
        # List current plan items
        runs = list(session.exec(
            select(PlanRun).where(PlanRun.user_id == user.id).order_by(PlanRun.generated_at.desc())
        ))
        if not runs:
            print("No plan runs found!")
            return
        
        run = runs[0]
        print(f"\nLatest PlanRun: ID={run.id}, status={run.status}")
        
        items = list(session.exec(
            select(PlanItem).where(PlanItem.plan_run_id == run.id).order_by(PlanItem.position)
        ))
        print(f"\nPlan has {len(items)} items:")
        for item in items:
            activity = session.get(Activity, item.activity_id)
            title = activity.title if activity else "UNKNOWN"
            print(f"  [{item.position}] {title[:50]}... (committed={item.committed})")
        
        # Try to find an activity by partial match
        if items:
            activity = session.get(Activity, items[0].activity_id)
            if activity:
                # Test with short fragment
                fragment = activity.title.split()[0] if activity.title else "test"
                print(f"\nTest: find_activity_by_title for '{fragment}'")
                found = find_activity_by_title(session, user, fragment)
                if found:
                    print(f"  FOUND: {found.title}")
                else:
                    print(f"  NOT FOUND!")
                
                # Test removal (dry run - just check if it would work)
                print(f"\nTest removal for: '{activity.title[:40]}'")
                result = remove_activity(session, user, activity.title)
                print(f"  Result: {result}")

if __name__ == "__main__":
    test_removal()
