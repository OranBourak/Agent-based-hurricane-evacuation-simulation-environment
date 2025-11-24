from dataclasses import dataclass
from typing import FrozenSet, Dict, List, Optional, Tuple
import heapq

from agent_base import Agent, Action, ActionType, Observation
from environment import Environment
from graph import Graph
from utils.heuristic_func import build_transformed_graph, heuristic

INF = 10**15
LIMIT = 10000


@dataclass(frozen=True)
class AStarState:
    current: int
    remaining: FrozenSet[int]
    equipped: bool


class AStarAgent(Agent):
    """A* search agent for evacuation planning."""

    def __init__(self, env: Environment, label="astar"):
        self.label = label
        self.base_graph = env.graph
        self.transformed_graph = build_transformed_graph(env.graph, env.P)
        self.expansions = 0

        self.action_array: List[Tuple[str, Optional[int]]] = []
        self.action_index = 0
        self.env = env

    def plan_route(self, start: int, obs: Observation) -> List[int]:
        """Plan route using A* search.
        Returns list of vertices in order to visit.
        """
        # reset expansions for this planning episode
        self.expansions = 0

        # initial state
        remaining = {
            vid for vid, v in self.transformed_graph.vertices.items()
            if v.people > 0 and vid != start
        }
        start_state = AStarState(start, frozenset(remaining), obs.self_state.equipped)

        # OPEN = priority queue
        OPEN: List[Tuple[float, float, int, AStarState]] = []
        counter = 0

        # G scores and parent pointers
        g_score: Dict[AStarState, float] = {start_state: 0.0}
        parent: Dict[AStarState, Optional[AStarState]] = {start_state: None}
        parent_move: Dict[AStarState, Optional[int]] = {start_state: None}

        # heuristic for start
        h0 = heuristic(self.transformed_graph, [start] + list(remaining))
        heapq.heappush(OPEN, (h0, 0.0, counter, start_state))

        # CLOSED stores best known g for each expanded state
        CLOSED: Dict[AStarState, float] = {}


        while OPEN and self.expansions < LIMIT:

            f, g, _, state = heapq.heappop(OPEN)
            self.expansions += 1  # count one expansion (we are expanding this state)

            # Check goal
            if not state.remaining:
                return self._reconstruct_route(state, parent, parent_move)

            # Graph-search duplicate detection (use g, not f)
            if state in CLOSED and CLOSED[state] <= g:
                continue
            CLOSED[state] = g

            # Expand neighbors
            for (n, _) in self.transformed_graph.adj[state.current]:

                # Cost of moving to neighbor using real rules
                g_step, act = self.base_graph.shortest_exec_path(
                    state.current, n, obs.Q, obs.U, obs.P, state.equipped
                )
                if g_step == INF:
                    continue

                # Compute the equipment AFTER the path
                new_equipped = state.equipped
                for kind, _ in act:
                    if kind == "equip":
                        new_equipped = True
                    elif kind == "unequip":
                        new_equipped = False

                g_new = g + g_step

                new_remaining = set(state.remaining)
                if n in new_remaining:
                    new_remaining.remove(n)
                new_state = AStarState(n, frozenset(new_remaining), new_equipped)

                # If we found a cheaper path to this state
                if g_new < g_score.get(new_state, INF):
                    g_score[new_state] = g_new
                    parent[new_state] = state
                    parent_move[new_state] = n

                    # Heuristic
                    h_new = heuristic(
                        self.transformed_graph,
                        [n] + list(new_remaining)
                    )
                    f_new = g_new + h_new

                    counter += 1
                    heapq.heappush(OPEN, (f_new, g_new, counter, new_state))

        # If we reached limit -> fail softly: return [start] (no movement)
        return [start]

    def _reconstruct_route(
        self,
        state: AStarState,
        parent: Dict[AStarState, Optional[AStarState]],
        parent_move: Dict[AStarState, Optional[int]]
    ) -> List[int]:
        """Reconstruct route from parent pointers."""

        route: List[int] = []
        cur: Optional[AStarState] = state

        while cur is not None:
            mv = parent_move[cur]
            if mv is not None:
                route.append(mv)
            cur = parent[cur]

        route.reverse()
        return route

    def _build_actions(self, obs: Observation):
        """Build the full low-level action list required to execute the A* route."""

        self.action_array = []
        self.action_index = 0

        start = obs.self_state.current_vertex
        equipped = obs.self_state.equipped

        # Ask A* to compute the HIGH-LEVEL route (list of vertices)
        route = self.plan_route(start, obs)

        # If route is empty
        if not route:
            return
        if len(route) == 1 and route[0] == start:
            return

        # Route returned by A* does NOT include the starting vertex.
        # We must chain shortest_exec_path(start -> route[0]) and so on.
        prev = start

        for v in route:
            # Compute optimal actions from prev -> v
            cost, actions = self.base_graph.shortest_exec_path(
                prev,
                v,
                obs.Q,
                obs.U,
                obs.P,
                equipped  # current equipment status
            )

            # Append actions to master action list
            self.action_array.extend(actions)

            # Update equipment state based on the actions we just appended
            for kind, _ in actions:
                if kind == "equip":
                    equipped = True
                elif kind == "unequip":
                    equipped = False

            # Move to next vertex in the route
            prev = v

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
