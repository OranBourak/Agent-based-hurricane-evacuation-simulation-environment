from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Tuple, Literal
import heapq

from agent_base import Agent, Action, ActionType, Observation
from environment import Environment
from graph import Graph, Edge
from utils.heuristic_func import build_transformed_graph, heuristic

INF = 10**15


@dataclass(frozen=True)
class RTAStarState:
    current: int
    remaining: FrozenSet[int]
    equipped: bool


class RealTimeAStarAgent(Agent):
    """
    Simplified Real-Time A*:
    - Each decide() runs at most L A* expansions, then commits to the first primitive action.
    - No offline full-plan execution unless you explicitly want caching.
    """

    def __init__(self, env: Environment, L: int = 10, label: str = "rt_astar") -> None:
        self.label = label
        self.L = int(L)
        self.base_graph: Graph = env.graph
        self.transformed_graph: Graph = build_transformed_graph(env.graph, env.P)

        # total expansions across the whole simulation
        self.expansions: int = 0

        # tiebreaker for heapq
        self._counter: int = 0

    def _remaining_from_obs(self, obs: Observation, at: int) -> FrozenSet[int]:
        return frozenset(
            vid for vid, (people, _kits) in obs.vertices.items()
            if people > 0 and vid != at
        )

    def _move_cost(self, edge: Edge, equipped: bool, obs: Observation) -> float:
        if edge.flooded and not equipped:
            return INF
        return edge.weight * (obs.P if equipped else 1)

    def _h(self, st: RTAStarState) -> float:
        return float(heuristic(self.transformed_graph, [st.current] + list(st.remaining)))

    def _reconstruct_actions(
        self,
        goal: RTAStarState,
        parent: Dict[RTAStarState, Optional[RTAStarState]],
        parent_action: Dict[RTAStarState, Optional[Tuple[str, Optional[int]]]],
    ) -> List[Tuple[str, Optional[int]]]:
        actions_rev: List[Tuple[str, Optional[int]]] = []
        cur: Optional[RTAStarState] = goal
        while cur is not None:
            act = parent_action.get(cur)
            if act is not None:
                actions_rev.append(act)
            cur = parent.get(cur)
        actions_rev.reverse()
        return actions_rev

    def _best_open_state(
        self,
        OPEN: List[Tuple[float, float, int, RTAStarState]],
        g_score: Dict[RTAStarState, float],
    ) -> Optional[RTAStarState]:
        best_state: Optional[RTAStarState] = None
        best_f = float("inf")
        for f, g, _c, st in OPEN:
            if g_score.get(st, INF) != g:
                continue  # stale entry
            if f < best_f:
                best_f = f
                best_state = st
        return best_state

    def _limited_astar(
        self, start: int, obs: Observation
    ) -> Tuple[Literal["goal", "partial", "fail"], List[Tuple[str, Optional[int]]]]:
        remaining = self._remaining_from_obs(obs, start)
        start_state = RTAStarState(start, remaining, obs.self_state.equipped)

        OPEN: List[Tuple[float, float, int, RTAStarState]] = []
        g_score: Dict[RTAStarState, float] = {start_state: 0.0}
        parent: Dict[RTAStarState, Optional[RTAStarState]] = {start_state: None}
        parent_action: Dict[RTAStarState, Optional[Tuple[str, Optional[int]]]] = {start_state: None}

        h0 = self._h(start_state)
        heapq.heappush(OPEN, (h0, 0.0, self._counter, start_state))
        self._counter += 1

        CLOSED: Dict[RTAStarState, float] = {}

        local_expansions = 0
        while OPEN and local_expansions < self.L:
            f, g, _c, state = heapq.heappop(OPEN)

            # stale entry
            if g_score.get(state, INF) != g:
                continue

            # count a real expansion (we are expanding this state now)
            local_expansions += 1
            self.expansions += 1

            # goal
            if not state.remaining:
                acts = self._reconstruct_actions(state, parent, parent_action)
                return "goal", acts

            # closed check by best g
            if state in CLOSED and CLOSED[state] <= g:
                continue
            CLOSED[state] = g

            u = state.current
            equipped = state.equipped
            kits_here = obs.vertices[u][1]

            # --------------------
            # SUCCESSORS = ONLY LEGAL PRIMITIVE ACTIONS
            # --------------------

            # EQUIP
            if (not equipped) and kits_here > 0:
                ns = RTAStarState(u, state.remaining, True)
                g_new = g + obs.Q
                if g_new < g_score.get(ns, INF):
                    g_score[ns] = g_new
                    parent[ns] = state
                    parent_action[ns] = ("equip", None)

                    f_new = g_new + self._h(ns)
                    heapq.heappush(OPEN, (f_new, g_new, self._counter, ns))
                    self._counter += 1

            # UNEQUIP
            if equipped:
                ns = RTAStarState(u, state.remaining, False)
                g_new = g + obs.U
                if g_new < g_score.get(ns, INF):
                    g_score[ns] = g_new
                    parent[ns] = state
                    parent_action[ns] = ("unequip", None)

                    f_new = g_new + self._h(ns)
                    heapq.heappush(OPEN, (f_new, g_new, self._counter, ns))
                    self._counter += 1

            # TRAVERSE one edge
            for v, edge in self.base_graph.neighbors(u):
                step = self._move_cost(edge, equipped, obs)
                if step == INF:
                    continue  # illegal

                new_remaining = set(state.remaining)
                new_remaining.discard(v)

                ns = RTAStarState(v, frozenset(new_remaining), equipped)
                g_new = g + step

                if g_new < g_score.get(ns, INF):
                    g_score[ns] = g_new
                    parent[ns] = state
                    parent_action[ns] = ("traverse", v)

                    f_new = g_new + self._h(ns)
                    heapq.heappush(OPEN, (f_new, g_new, self._counter, ns))
                    self._counter += 1

        # If we didn't reach a goal within L expansions, pick best OPEN state and commit to its first action
        best = self._best_open_state(OPEN, g_score)
        if best is None:
            return "fail", []

        acts = self._reconstruct_actions(best, parent, parent_action)
        return "partial", acts

    def _tuple_to_action(self, tup: Tuple[str, Optional[int]]) -> Action:
        kind, arg = tup
        if kind == "traverse":
            return Action(ActionType.TRAVERSE, arg)
        if kind == "equip":
            return Action(ActionType.EQUIP)
        if kind == "unequip":
            return Action(ActionType.UNEQUIP)
        return Action(ActionType.NO_OP)

    def decide(self, obs: Observation) -> Action:
        start = obs.self_state.current_vertex
        status, acts = self._limited_astar(start, obs)

        if status in ("goal", "partial") and acts:
            # commit to FIRST primitive action only
            return self._tuple_to_action(acts[0])

        return Action(ActionType.NO_OP)