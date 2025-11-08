from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Protocol

# ------------------ Core interfaces ------------------

class ActionType(Enum):
    ''' Types of actions an agent can take '''
    #auto() assigns unique values automatically
    TRAVERSE = auto() 
    EQUIP = auto()
    UNEQUIP = auto()
    NO_OP = auto()
    TERMINATE = auto()

@dataclass(frozen=True)
class Action:
    ''' Represents an action taken by an agent

    Attributes:
        action.kind: the type of action
        action.to_vertex: used only for TRAVERSE actions, otherwise None'''
    kind: ActionType
    to_vertex: Optional[int] = None # used only for TRAVERSE actions, otherwise None (therefore optional)

@dataclass(frozen=True)
class AgentState:
    agent_id: int
    label: str
    current_vertex: int
    equipped: bool
    rescued: int
    actions_done: int = 0

@dataclass(frozen=True)
class Observation:
    """Full world snapshot given to each agent.
    Used as input to the agent's decide() method."""

    time: int
    Q: int
    U: int
    P: int
    vertices: Dict[int, Tuple[int, bool]]
    edges: List[Tuple[int, int, int, bool]]
    agents: List[AgentState]
    self_state: AgentState

class Agent(Protocol):
    """Base class for all agents.

    Attributes:
        label (str): Label of the agent.
        can_rescue (bool): Whether the agent can rescue people.

    Methods:
        decide(obs): Given an Observation, return an Action.
    """
    label: str
    can_rescue: bool
    def decide(self, obs: Observation) -> Action: ...
