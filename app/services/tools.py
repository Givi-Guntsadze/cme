import json
import logging
from typing import Any, Dict, List, Optional
from sqlmodel import Session
from app.models import User, Activity
from app.services.discovery import find_activity_by_title
from app.services.plan import PlanManager, load_policy_bundle, apply_policy_payloads

logger = logging.getLogger(__name__)

# --- Tool Schemas ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "update_activity_status",
            "description": "Update the cost or eligibility status of an activity. Use this when the user says 'mark X as eligible' or 'X costs $500'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_title": {
                        "type": "string",
                        "description": "The name or title of the activity to update."
                    },
                    "cost": {
                        "type": "number",
                        "description": "New cost in USD, if provided."
                    },
                    "eligible": {
                        "type": "boolean",
                        "description": "Set to true if user confirms eligibility."
                    }
                },
                "required": ["activity_title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_activity",
            "description": "Remove an activity from the user's plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_title": {
                        "type": "string",
                        "description": "The name of the activity to remove."
                    }
                },
                "required": ["activity_title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_activity",
            "description": "Add a specific activity to the plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_title": {
                        "type": "string",
                        "description": "The name of the activity to add."
                    }
                },
                "required": ["activity_title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_activities",
            "description": "Search for new activities based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search terms (e.g., 'patient safety', 'online courses')."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_preferences",
            "description": "Update user preferences like budget or modality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {
                        "type": "number",
                        "description": "Maximum budget in dollars."
                    },
                    "days_off": {
                        "type": "number",
                        "description": "Number of days off work."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mark_activity_complete",
            "description": "Mark an activity as completed/done (log it).",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_title": {
                        "type": "string",
                        "description": "The name of the activity to mark as complete."
                    },
                    "credits": {
                        "type": "number",
                        "description": "Credits earned, if specified."
                    }
                },
                "required": ["activity_title"]
            }
        }
    },
]

# --- Tool Implementations ---

def update_activity_status(session: Session, user: User, activity_title: str, cost: Optional[float] = None, eligible: Optional[bool] = None) -> str:
    activity = find_activity_by_title(session, user, activity_title)
    if not activity:
        return f"Error: Could not find activity matching '{activity_title}'."
    
    updates = []
    if cost is not None:
        activity.cost_usd = cost
        updates.append(f"cost set to ${cost}")
    
    if eligible is True:
        # Clear ALL eligibility restriction fields
        # This ensures the missing_profile_data check won't trigger 'uncertain' status
        activity.eligibility_text = None
        activity.open_to_public = True
        activity.eligible_institutions = None  # Clear institution requirements
        activity.eligible_groups = None        # Clear group requirements  
        activity.membership_required = None    # Clear membership requirements
        updates.append("marked as eligible")
        
        # Also update PlanItem.eligibility_status (this is what the UI displays)
        from sqlmodel import select
        from app.models import PlanItem, PlanRun
        # Find all PlanItems for this activity in user's plans
        plan_items = session.exec(
            select(PlanItem)
            .join(PlanRun)
            .where(PlanRun.user_id == user.id, PlanItem.activity_id == activity.id)
        ).all()
        logger.info(f"Found {len(plan_items)} PlanItems for activity {activity.id}")
        for pi in plan_items:
            logger.info(f"Updating PlanItem {pi.id} eligibility_status from '{pi.eligibility_status}' to 'eligible'")
            pi.eligibility_status = "eligible"
            session.add(pi)
    
    if not updates:
        return f"Found '{activity.title}' but no changes were requested."
    
    session.add(activity)
    session.commit()
    
    # DEBUG: Verify the commit persisted correctly
    session.refresh(activity)
    logger.info(f"[ELIGIBILITY DEBUG] After commit: activity.id={activity.id}, open_to_public={activity.open_to_public}, eligibility_text='{activity.eligibility_text}'")
    
    # ALWAYS invalidate plans after updates to force regeneration with fresh data
    # This ensures the plan is rebuilt using the updated activity.open_to_public value
    PlanManager.invalidate_user_plans(session, user.id, reason="activity_status_update")
    
    return f"Updated '{activity.title}': {', '.join(updates)}."

def remove_activity(session: Session, user: User, activity_title: str) -> str:
    logger.info(f"[REMOVE_ACTIVITY] Called with title='{activity_title}' for user={user.id}")
    activity = find_activity_by_title(session, user, activity_title)
    if not activity:
        logger.warning(f"[REMOVE_ACTIVITY] Activity not found for '{activity_title}'")
        return f"Error: Could not find activity matching '{activity_title}' to remove."
    
    logger.info(f"[REMOVE_ACTIVITY] Found activity: {activity.title} (ID={activity.id})")
    
    # Step 1: Find and uncommit/remove any existing PlanItems for this activity
    from sqlmodel import select
    from app.models import PlanItem, PlanRun
    
    plan_items = list(session.exec(
        select(PlanItem)
        .join(PlanRun)
        .where(
            PlanRun.user_id == user.id,
            PlanRun.status == "active",
            PlanItem.activity_id == activity.id
        )
    ))
    
    for pi in plan_items:
        logger.info(f"[REMOVE_ACTIVITY] Removing PlanItem {pi.id} for activity {activity.id}")
        session.delete(pi)
    
    if plan_items:
        session.flush()
    
    # Step 2: Apply policy to prevent activity from being re-added during plan regeneration
    # Create policy for BOTH 'default' and 'balanced' modes to ensure coverage
    removal_payload = json.dumps({
        "by_mode": {
            "balanced": {"remove_titles": [activity.title]},
            "cheapest": {"remove_titles": [activity.title]},
        },
        "remove_titles": [activity.title]  # Also as default fallback
    })
    logger.info(f"[REMOVE_ACTIVITY] Applying policy payload: {removal_payload}")
    apply_policy_payloads([removal_payload], user, session, invalidate=False)
    
    # Step 3: Invalidate plans to force regeneration
    logger.info(f"[REMOVE_ACTIVITY] Invalidating user plans...")
    PlanManager.invalidate_user_plans(session, user.id, reason="remove_activity")
    session.commit()
    logger.info(f"[REMOVE_ACTIVITY] Successfully removed '{activity.title}'")
    
    return f"Removed '{activity.title}' from your plan."

def add_activity(session: Session, user: User, activity_title: str) -> str:
    # Check if it exists
    activity = find_activity_by_title(session, user, activity_title)
    if not activity:
        return f"Error: Could not find activity matching '{activity_title}'. Try searching first."
    
    cost = f"${activity.cost_usd}" if activity.cost_usd else "Free"
    return f"Found '{activity.title}' ({cost}). Use normal chat to confirm adding it."


def update_preferences(session: Session, user: User, budget: Optional[float] = None, days_off: Optional[float] = None) -> str:
    updates = []
    if budget is not None:
        user.budget_usd = budget
        updates.append(f"Budget set to ${budget}")
    if days_off is not None:
        user.days_off = int(days_off)
        updates.append(f"Days off set to {int(days_off)}")
    
    if not updates:
        return "No preference changes requested."
    
    session.add(user)
    session.commit()
    PlanManager.invalidate_user_plans(session, user.id, reason="preference_update")
    return f"Preferences updated: {', '.join(updates)}."

def search_activities(query: str) -> str:
    try:
        from app.services import vectordb
        results = vectordb.search(query, n_results=5)
        if not results:
            return f"No activities found matching '{query}'."
        
        lines = [f"Found {len(results)} matches for '{query}':"]
        for r in results:
            cost = f"${r['cost']}" if r.get('cost') else "Free/Unknown"
            lines.append(f"- {r['title']} ({r['provider']}): {r['credits']} cr, {cost}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return "Error occurred during search."

def mark_activity_complete(session: Session, user: User, activity_title: str, credits: Optional[float] = None) -> str:
    """Mark an activity as completed. Creates a Claim, updates user credits, and archives the activity."""
    from datetime import date as date_type
    from app.models import Claim, CompletedActivity, PlanItem, PlanRun
    from sqlmodel import select
    
    logger.info(f"[COMPLETE] Called for '{activity_title}' by user {user.id}")
    
    activity = find_activity_by_title(session, user, activity_title)
    if not activity:
        return f"Error: Could not find activity matching '{activity_title}'."
    
    earned_credits = credits if credits is not None else (activity.credits or 0.0)
    logger.info(f"[COMPLETE] Found activity: {activity.title} (ID={activity.id}), credits={earned_credits}")
    
    # Step 1: Create Claim record (required fields: user_id, credits, date, source_text)
    claim = Claim(
        user_id=user.id,
        credits=earned_credits,
        topic=activity.title[:100] if activity.title else "CME Activity",
        date=date_type.today(),
        source_text=f"{activity.provider or 'Unknown'}: {activity.title or 'Activity'}"[:200],
    )
    session.add(claim)
    logger.info(f"[COMPLETE] Created Claim for {earned_credits} credits")
    
    # Step 2: Create CompletedActivity record to prevent it from appearing in future plans
    completed = CompletedActivity(
        user_id=user.id,
        activity_id=activity.id,
    )
    session.add(completed)
    logger.info(f"[COMPLETE] Created CompletedActivity record")
    
    # Step 3: Remove any PlanItems for this activity from active plans
    plan_items = list(session.exec(
        select(PlanItem)
        .join(PlanRun)
        .where(
            PlanRun.user_id == user.id,
            PlanRun.status == "active",
            PlanItem.activity_id == activity.id
        )
    ))
    for pi in plan_items:
        logger.info(f"[COMPLETE] Removing PlanItem {pi.id}")
        session.delete(pi)
    
    # Step 4: Update user's remaining credits
    user.remaining_credits = max((user.remaining_credits or 0.0) - earned_credits, 0.0)
    session.add(user)
    logger.info(f"[COMPLETE] Updated user.remaining_credits to {user.remaining_credits}")
    
    # Step 5: Invalidate plans to trigger regeneration
    PlanManager.invalidate_user_plans(session, user.id, reason="activity_completed")
    session.commit()
    logger.info(f"[COMPLETE] Successfully marked '{activity.title}' as complete")
    
    return f"Marked '{activity.title}' as complete ({earned_credits:.1f} credits earned). Your remaining credits needed: {user.remaining_credits:.1f}."


def execute_tool_call(tool_name: str, args: Dict[str, Any], session: Session, user: User) -> str:
    """Dispatcher for tool calls."""
    if tool_name == "update_activity_status":
        return update_activity_status(
            session, 
            user, 
            args.get("activity_title"), 
            cost=args.get("cost"), 
            eligible=args.get("eligible")
        )
    elif tool_name == "remove_activity":
        return remove_activity(session, user, args.get("activity_title"))
    elif tool_name == "add_activity":
        return add_activity(session, user, args.get("activity_title"))
    elif tool_name == "update_preferences":
        return update_preferences(
            session, 
            user, 
            budget=args.get("budget"), 
            days_off=args.get("days_off")
        )
    elif tool_name == "search_activities":
        return search_activities(args.get("query"))
    elif tool_name == "mark_activity_complete":
        return mark_activity_complete(
            session,
            user,
            args.get("activity_title"),
            credits=args.get("credits")
        )
    else:
        return f"Error: Unknown tool '{tool_name}'."

