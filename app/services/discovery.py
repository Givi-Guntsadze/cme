import re
from difflib import SequenceMatcher
from sqlmodel import select, Session
from app.models import Activity, User, PlanItem
from app.services.plan import PlanManager, load_policy_bundle

TITLE_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
TITLE_STOPWORDS = {
    "the", "a", "an", "with", "for", "and", "event", "conference",
    "meeting", "cme", "annual", "update", "institute", "activity",
    "course", "plan", "session", "workshop", "hybrid", "primary",
}
MIN_MATCH_SCORE = 0.25

def _title_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in TITLE_TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        if not token:
            continue
        if token in TITLE_STOPWORDS and len(token) < 5:
            continue
        if token.isdigit() and len(token) < 4:
            continue
        tokens.add(token)
    return tokens

def _match_score(search: str, title: str) -> float:
    title_lower = (title or "").lower()
    if not title_lower:
        return 0.0
    ratio = SequenceMatcher(None, search, title_lower).ratio()
    if search in title_lower:
        ratio += 0.75
    if title_lower.startswith(search):
        ratio += 0.25
    tokens = [t for t in search.split(" ") if t]
    if tokens:
        hits = sum(1 for token in tokens if token in title_lower)
        ratio += 0.1 * hits
    return ratio

def _is_candidate_match(candidate_title: str, query_tokens: set[str]) -> bool:
    if not candidate_title or not query_tokens:
        return False
    candidate_tokens = _title_tokens(candidate_title)
    if not candidate_tokens:
        return False
    overlap = query_tokens & candidate_tokens
    if not overlap:
        return False
    if len(query_tokens) <= 3:
        return overlap == query_tokens
    overlap_threshold = max(2, len(query_tokens) // 2)
    return len(overlap) >= overlap_threshold

def find_activity_by_title(session: Session, user: User, text: str) -> Activity | None:
    search = (text or "").strip().lower()
    if not search:
        return None

    plan_manager = PlanManager(session)
    policy_bundle = load_policy_bundle(session, user)
    # Ensure plan exists to check against plan items first
    run = plan_manager.ensure_plan(user, "balanced", policy_bundle)

    candidates: list[tuple[int, float, int, Activity]] = []
    seen_ids: set[int] = set()

    if run:
        plan_items = list(
            session.exec(
                select(PlanItem)
                .where(PlanItem.plan_run_id == run.id)
                .order_by(PlanItem.position.asc())
            )
        )
        for item in plan_items:
            activity = session.get(Activity, item.activity_id)
            if not activity or not activity.title:
                continue
            score = _match_score(search, activity.title)
            if score < MIN_MATCH_SCORE:
                continue
            priority = 0 if item.committed else 1
            candidates.append(
                (
                    priority,
                    -score,
                    item.position,
                    activity.id or 0,
                    activity,
                )
            )
            seen_ids.add(activity.id)

    # Also search catalog
    catalog_stmt = select(Activity)
    for activity in session.exec(catalog_stmt):
        if activity.id in seen_ids:
            continue
        if not activity.title:
            continue
        score = _match_score(search, activity.title)
        if score < MIN_MATCH_SCORE:
            continue
        candidates.append(
            (
                2,
                -score,
                len(activity.title or ""),
                activity.id or 0,
                activity,
            )
        )

    if not candidates:
        return None

    candidates.sort()
    best_activity = candidates[0][4]
    best_score = -candidates[0][1]
    if best_score < MIN_MATCH_SCORE:
        return None
    return best_activity
