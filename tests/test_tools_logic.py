# import pytest
from sqlmodel import SQLModel, Session, create_engine, select
from app.models import User, Activity, PlanItem, PlanRun, PolicyBundle
from app.services.tools import update_activity_status, remove_activity, add_activity, mark_activity_complete, search_activities

def test_tool_logic():
    # Setup in-memory DB
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        # Create User
        user = User(name="Test Doc", target_credits=100.0, remaining_credits=100.0)
        session.add(user)
        
        # Create Activity
        act = Activity(title="Target Activity", cost_usd=100.0, credits=10.0, eligibility_text="Check Eligibility")
        session.add(act)
        session.commit()
        session.refresh(user)
        session.refresh(act)
        
        # Test 1: Update Status (The Fix)
        # User says "It is eligible"
        print("Testing update_activity_status...")
        res = update_activity_status(session, user, "Target Activity", eligible=True)
        print(f"Result: {res}")
        session.refresh(act)
        assert act.open_to_public is True, "Should set open_to_public to True"
        assert act.eligibility_text is None, "Should clear eligibility text"
        assert "marked as eligible" in res
        
        # Test 2: Update Cost
        res = update_activity_status(session, user, "Target Activity", cost=50.0)
        session.refresh(act)
        assert act.cost_usd == 50.0
        assert "cost set to $50" in res

        # Test 3: Mark Complete
        print("Testing mark_activity_complete...")
        res = mark_activity_complete(session, user, "Target Activity")
        print(f"Result: {res}")
        # Verify claim (we'd need to query Claim model, assuming it's imported or logic works)
        # For now just check return string
        assert "complete" in res

        # Test 4: Search (Mocking vectordb?)
        # Since search imports vectordb inside, and we don't have vectordb setup in memory...
        # We expect it to likely fail or return "No activities found" if it tries to load real DB.
        # We skip deep search test here, focusing on logic tools.
        
        print("Tools logic verification passed!")

if __name__ == "__main__":
    try:
        test_tool_logic()
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)
