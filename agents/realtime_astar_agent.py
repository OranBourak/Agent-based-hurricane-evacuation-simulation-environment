# realtime_astar_agent.py  (put next to astar_agent.py)
from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Tuple, Literal
import heapq

from agent_base import Agent, Action, ActionType, Observation
from environment import Environment
from graph import Graph
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
    - Each decide() runs at most L A* expansions, then commits to the first low-level action.
    - If it finds a complete optimal plan, it caches it and then does 0 expansions per move.
    """

    def __init__(self, env: Environment, L: int = 10, label: str = "rt_astar") -> None:
        self.label = label
        self.L = int(L)
        self.base_graph: Graph = env.graph
        self.transformed_graph: Graph = build_transformed_graph(env.graph, env.P)

        # total expansions across the whole simulation
        self.expansions: int = 0

        # cached full plan once solved
        self._solved_plan: bool = False
        self.action_array: List[Tuple[str, Optional[int]]] = []
        self.action_index: int = 0

    def _remaining_from_obs(self, obs: Observation, start: int) -> FrozenSet[int]:

        return frozenset(
            vid for vid, (people, _kits) in obs.vertices.items()
            if people > 0 and vid != start
        )

    def _reconstruct_route(
        self,
        state: RTAStarState,
        parent: Dict[RTAStarState, Optional[RTAStarState]],
        parent_move: Dict[RTAStarState, Optional[int]],
    ) -> List[int]:
        route: List[int] = []
        cur: Optional[RTAStarState] = state
        while cur is not None:
            mv = parent_move.get(cur)
            if mv is not None:
                route.append(mv)
            cur = parent.get(cur)
        route.reverse()
        return route

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
    ) -> Tuple[Literal["goal", "partial", "fail"], List[int]]:
        remaining = self._remaining_from_obs(obs, start)
        start_state = RTAStarState(start, remaining, obs.self_state.equipped)

        OPEN: List[Tuple[float, float, int, RTAStarState]] = []
        counter = 0

        g_score: Dict[RTAStarState, float] = {start_state: 0.0}
        parent: Dict[RTAStarState, Optional[RTAStarState]] = {start_state: None}
        parent_move: Dict[RTAStarState, Optional[int]] = {start_state: None}

        h0 = heuristic(self.transformed_graph, [start] + list(remaining))
        heapq.heappush(OPEN, (float(h0), 0.0, counter, start_state))

        CLOSED: Dict[RTAStarState, float] = {}

        local_expansions = 0
        while OPEN and local_expansions < self.L:
            f, g, _c, state = heapq.heappop(OPEN)

            # stale queue entry
            if g_score.get(state, INF) != g:
                continue

            local_expansions += 1
            self.expansions += 1

            if not state.remaining:
                return "goal", self._reconstruct_route(state, parent, parent_move)

            if state in CLOSED and CLOSED[state] <= g:
                continue
            CLOSED[state] = g

            for (n, _e) in self.transformed_graph.adj[state.current]:
                g_step, act_list = self.base_graph.shortest_exec_path(
                    state.current, n, obs.Q, obs.U, obs.P, state.equipped
                )
                if g_step == INF:
                    continue

                new_equipped = state.equipped
                for kind, _arg in act_list:
                    if kind == "equip":
                        new_equipped = True
                    elif kind == "unequip":
                        new_equipped = False

                g_new = g + float(g_step)

                new_remaining = set(state.remaining)
                new_remaining.discard(n)
                new_state = RTAStarState(n, frozenset(new_remaining), new_equipped)

                if g_new < g_score.get(new_state, INF):
                    g_score[new_state] = g_new
                    parent[new_state] = state
                    parent_move[new_state] = n

                    h_new = heuristic(self.transformed_graph, [n] + list(new_remaining))
                    f_new = g_new + float(h_new)

                    counter += 1
                    heapq.heappush(OPEN, (f_new, g_new, counter, new_state))

        best = self._best_open_state(OPEN, g_score)
        if best is None:
            return "fail", []
        return "partial", self._reconstruct_route(best, parent, parent_move)

    def _build_full_action_plan(self, start: int, obs: Observation, route: List[int]) -> None:
        self.action_array = []
        self.action_index = 0

        equipped = obs.self_state.equipped
        prev = start
        for v in route:
            cost, actions = self.base_graph.shortest_exec_path(prev, v, obs.Q, obs.U, obs.P, equipped)
            if cost == INF:
                break
            self.action_array.extend(actions)
            for kind, _arg in actions:
                if kind == "equip":
                    equipped = True
                elif kind == "unequip":
                    equipped = False
            prev = v

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
        # Already solved -> just execute (0 expansions per move)
        if self._solved_plan:
            if self.action_index >= len(self.action_array):
                return Action(ActionType.NO_OP)
            act = self._tuple_to_action(self.action_array[self.action_index])
            self.action_index += 1
            return act

        start = obs.self_state.current_vertex
        status, route = self._limited_astar(start, obs)

        if status == "goal":
            self._build_full_action_plan(start, obs, route)
            self._solved_plan = True
            if not self.action_array:
                return Action(ActionType.NO_OP)
            act = self._tuple_to_action(self.action_array[0])
            self.action_index = 1
            return act

        if status == "partial" and route:
            # commit only to the first low-level action
            next_vertex = route[0]
            cost, actions = self.base_graph.shortest_exec_path(
                start, next_vertex, obs.Q, obs.U, obs.P, obs.self_state.equipped
            )
            if cost == INF or not actions:
                return Action(ActionType.NO_OP)
            return self._tuple_to_action(actions[0])

        return Action(ActionType.NO_OP)
