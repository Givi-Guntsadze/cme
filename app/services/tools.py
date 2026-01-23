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
        activity.eligibility_text = None
        activity.open_to_public = True
        updates.append("marked as eligible")
    
    if not updates:
        return f"Found '{activity.title}' but no changes were requested."
    
    session.add(activity)
    session.commit()
    
    # Invalidate plan to reflect changes
    PlanManager.invalidate_user_plans(session, user.id, reason="activity_update")
    
    return f"Updated '{activity.title}': {', '.join(updates)}."

def remove_activity(session: Session, user: User, activity_title: str) -> str:
    activity = find_activity_by_title(session, user, activity_title)
    if not activity:
        return f"Error: Could not find activity matching '{activity_title}' to remove."
    
    # Logic similar to main.py remove handler
    policy_bundle = load_policy_bundle(session, user)
    plan_manager = PlanManager(session)
    # Ensure raw plan exists so we can uncommit items if needed
    plan_manager.ensure_plan(user, "balanced", policy_bundle)
    
    # We use the policy payload mechanism to persist removal
    removal_payload = json.dumps({"remove_titles": [activity.title]})
    apply_policy_payloads([removal_payload], user, session, invalidate=False)
    
    PlanManager.invalidate_user_plans(session, user.id, reason="remove_activity")
    session.commit()
    
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
    activity = find_activity_by_title(session, user, activity_title)
    if not activity:
        return f"Error: Could not find activity matching '{activity_title}'."
    
    # Simplistic logic: Just log it. In real app, we might check Claim table.
    from app.models import Claim, CompletedActivity
    
    # Create claim
    claim = Claim(
        user_id=user.id,
        activity_id=activity.id,
        credits=credits or activity.credits or 0.0,
        date=None # Today?
    )
    session.add(claim)
    
    # Also mark as completed behavior (invalidate plan)
    PlanManager.invalidate_user_plans(session, user.id, reason="activity_completed")
    session.commit()
    
    return f"Marked '{activity.title}' as complete ({claim.credits} cr)."


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

