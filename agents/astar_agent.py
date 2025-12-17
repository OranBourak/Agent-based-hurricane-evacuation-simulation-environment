from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Tuple
import heapq

from agent_base import Agent, Action, ActionType, Observation
from environment import Environment
from graph import Edge
from utils.heuristic_func import build_transformed_graph, heuristic

INF = 10**15
LIMIT = 10000


@dataclass(frozen=True)
class AStarState:
    current: int
    remaining: FrozenSet[int]
    equipped: bool


class AStarAgent(Agent):
    def __init__(self, env: Environment, label="astar"):
        self.label = label
        self.base_graph = env.graph
        self.transformed_graph = build_transformed_graph(env.graph, env.P)

        self.expansions = 0
        self.action_array: List[Tuple[str, Optional[int]]] = []
        self.action_index = 0
        self.env = env

    # ------------------------------------------------------------
    # A* SEARCH
    # ------------------------------------------------------------
    def plan_route(self, start: int, obs: Observation) -> List[AStarState]:
        """Run standard A* and return a list of states (path)."""

        self.expansions = 0
        self._counter = 0


        remaining = {
            vid for vid, (ppl, _) in obs.vertices.items()
            if ppl > 0 and vid != start
        }

        start_state = AStarState(start, frozenset(remaining), obs.self_state.equipped)

        OPEN: List[Tuple[float, float, int, AStarState]] = []  # (f, g, counter, state)
        counter = 0

        g_score: Dict[AStarState, float] = {start_state: 0.0}
        parent: Dict[AStarState, Optional[AStarState]] = {start_state: None}
        parent_action: Dict[AStarState, Optional[Tuple[str, Optional[int]]]] = {
            start_state: None
        } # Initial action is None, Optional for actions: ("traverse", v), ("equip", None), ("unequip", None)

        h0 = heuristic(self.transformed_graph, [start] + list(remaining))
        heapq.heappush(OPEN, (h0, 0.0, self._counter, start_state))

        CLOSED: Dict[AStarState, float] = {}

        while OPEN and self.expansions < LIMIT:
            f, g, _, state = heapq.heappop(OPEN)
            self.expansions += 1  # expansion

            if not state.remaining:
                return self._reconstruct_states(state, parent)

            if state in CLOSED and CLOSED[state] <= g:
                continue
            CLOSED[state] = g

            u = state.current
            equipped = state.equipped
            kits_at_u = obs.vertices[u][1]

            # --------------------
            # EQUIP
            # --------------------
            if not equipped and kits_at_u > 0:
                ns = AStarState(u, state.remaining, True)
                g_new = g + obs.Q
                self._push_state(state, ns, ("equip", None),
                                 g_new, g_score, parent, parent_action,
                                 obs, OPEN)

            # --------------------
            # UNEQUIP
            # --------------------
            if equipped:
                ns = AStarState(u, state.remaining, False)
                g_new = g + obs.U
                self._push_state(state, ns, ("unequip", None),
                                 g_new, g_score, parent, parent_action,
                                 obs, OPEN)

            # --------------------
            # TRAVERSE
            # --------------------
            for v, edge in self.base_graph.neighbors(u):
                move_cost = self._move_cost(edge, equipped, obs)
                if move_cost == INF:
                    continue

                new_remaining = set(state.remaining)
                new_remaining.discard(v)

                ns = AStarState(v, frozenset(new_remaining), equipped)
                g_new = g + move_cost

                self._push_state(state, ns, ("traverse", v),
                                 g_new, g_score, parent, parent_action,
                                 obs, OPEN)

        return []

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def _move_cost(self, edge: Edge, equipped: bool, obs: Observation) -> float:
        if edge.flooded and not equipped:
            return INF
        if equipped:
            return edge.weight * obs.P
        return edge.weight

    def _push_state(
        self,
        state: AStarState,
        next_state: AStarState,
        action: Tuple[str, Optional[int]],
        g_new: float,
        g_score,
        parent,
        parent_action,
        obs,
        OPEN,
    ):
        if g_new >= g_score.get(next_state, INF):
            return

        g_score[next_state] = g_new
        parent[next_state] = state
        parent_action[next_state] = action

        h = heuristic(self.transformed_graph,
                      [next_state.current] + list(next_state.remaining))
        f = g_new + h
        self._counter += 1 #needed to avoid comparison of AStarState in heapq when 2 states have same f  score
        heapq.heappush(OPEN, (f, g_new, self._counter, next_state))

    def _reconstruct_states(
        self,
        goal: AStarState,
        parent: Dict[AStarState, Optional[AStarState]]
    ) -> List[AStarState]:
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    # ------------------------------------------------------------
    # ACTION GENERATION
    # ------------------------------------------------------------
    def _build_actions(self, obs: Observation):
        self.action_array = []
        self.action_index = 0

        path = self.plan_route(obs.self_state.current_vertex, obs)
        if not path:
            return

        for i in range(1, len(path)):
            prev = path[i - 1]
            cur = path[i]

            if prev.current == cur.current:
                if not prev.equipped and cur.equipped:
                    self.action_array.append(("equip", None))
                elif prev.equipped and not cur.equipped:
                    self.action_array.append(("unequip", None))
            else:
                self.action_array.append(("traverse", cur.current))

    # ------------------------------------------------------------
    # DECISION
    # ------------------------------------------------------------
    def decide(self, obs: Observation) -> Action:
        if self.action_index == 0:
            self._build_actions(obs)

        if self.action_index >= len(self.action_array):
            return Action(ActionType.NO_OP)

        kind, arg = self.action_array[self.action_index]
        self.action_index += 1

        if kind == "traverse":
            return Action(ActionType.TRAVERSE, arg)
        if kind == "equip":
            return Action(ActionType.EQUIP)
        if kind == "unequip":
            return Action(ActionType.UNEQUIP)
        return Action(ActionType.NO_OP)
