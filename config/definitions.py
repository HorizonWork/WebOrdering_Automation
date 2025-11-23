
"""
Python Type Definitions for WebOrdering Automation Data Schemas.
Matches the JSON schemas in config/schemas/.
"""

from typing import TypedDict, List, Dict, Any, Optional

# ------------------------------------------------------------------------------
# PLANNER SCHEMA
# ------------------------------------------------------------------------------

class PlannerGoal(TypedDict):
    raw_user_goal: str
    constraints: Dict[str, Any]

class PlannerHighLevelState(TypedDict):
    page_type: str
    current_step: str
    has_price_filter: bool
    has_official_store_filter: bool

class PlannerInput(TypedDict):
    goal: PlannerGoal
    high_level_state: PlannerHighLevelState
    history_summary: str
    visual_summary: str
    detected_obstacles: List[str]
    available_high_level_actions: List[str]

class PlanStep(TypedDict):
    reasoning: str
    description: str

class PlannerOutput(TypedDict):
    plan_version: int
    overall_strategy: str
    next_plan_step: PlanStep

# ------------------------------------------------------------------------------
# CONTROLLER SCHEMA
# ------------------------------------------------------------------------------

class ControllerGoal(TypedDict):
    summary: str

class ControllerCurrentPlanStep(TypedDict):
    type: str
    description: str
    constraints: Dict[str, Any]

class ControllerPageState(TypedDict):
    page_type: str
    dom_state: Dict[str, Any]
    vision_state: Dict[str, Any]

class ControllerAction(TypedDict):
    type: str  # CLICK, FILL, SCROLL, WAIT, REQUEST_HELP
    id: str    # CSS Selector or unique ID
    description: str
    text: Optional[str] # For FILL/TYPE
    params: Optional[Dict[str, Any]] # Full parameters for execution

class ControllerInput(TypedDict):
    goal: ControllerGoal
    current_plan_step: ControllerCurrentPlanStep
    page_state: ControllerPageState
    available_actions_flat: List[Dict[str, Any]]
    last_action_result: str
    short_history: str

class ControllerOutput(TypedDict):
    chosen_action: ControllerAction
    reason: str

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

AVAILABLE_HIGH_LEVEL_ACTIONS = [
    "SEARCH_PRODUCT",
    "APPLY_FILTER",
    "SELECT_PRODUCT",
    "GO_TO_CART",
    "GO_TO_CHECKOUT",
    "FILL_CHECKOUT_INFO",
    "REVIEW_ORDER",
    "TERMINATE",
]

ACTION_TYPES = [
    "CLICK",
    "FILL",
    "SCROLL",
    "WAIT",
    "REQUEST_HELP",
]
