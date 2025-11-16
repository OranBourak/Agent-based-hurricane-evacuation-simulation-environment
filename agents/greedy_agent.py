from dataclasses import dataclass
from typing import List, Set, Dict
from agent_base import Agent, Observation, Action, ActionType
from environment import Environment
from graph import Graph
from utils.heuristic_func import build_transformed_graph, heuristic


@dataclass
class SearchState:
    """
    A node in the abstract search tree.

    For the greedy agent we only really use:
      - current: where we stand now
      - remaining: which interesting vertices still have people
      - route: the order in which we've visited interesting vertices so far
    """
    current: int
    remaining: Set[int]
    route: List[int]
    # expansions: int


class GreedyAgent(Agent):
    """
    Part-2 greedy *search* agent (NOT the stupid greedy from part 1).

    It assumes:
      - it acts alone
      - the heuristic world uses the transformed graph
        (kits ignored, flooded costs multiplied by P).

    Interface here is a pure planner:
      plan_route(start_vertex) -> [v1, v2, ...]
    """

    def __init__(self, env: Environment, label: str = "greedy") -> None:
        self.label = label
        # 1) keep original graph if you ever need it
        self.base_graph = env.graph
        # 2) build and keep the transformed graph (only ONCE per agent)
        self.transformed_graph = build_transformed_graph(env.graph, env.P)
        self.env = env
        self.action_index = 0
        self.expansions = 0

    # ---------- core greedy search ----------

    def _initial_state(self, start_vertex: int) -> SearchState:
        """Create initial search state from start vertex."""
        # "Interesting" nodes = vertices that currently have people
        remaining = {
            vid for vid, v in self.transformed_graph.vertices.items()
            if v.people > 0 and vid != start_vertex
        }

        route: List[int] = []
        # If start itself has people, we conceptually "visit" it first
        if self.transformed_graph.vertices[start_vertex].people > 0:
            route.append(start_vertex)

        return SearchState(
            current=start_vertex,
            remaining=remaining,
            route=route,
        )

    def _choose_best_successor(self, state: SearchState) -> int | None:
        """
        Given a search state, pick the 'best' next vertex from remaining,
        according to the heuristic (distance on transformed graph).

        This is the "expand node with lowest heuristic value" step.
        """
        best_v = None
        best_h = float("inf")

        for v in state.remaining:
            h = heuristic(state.current, v, self.transformed_graph)
            if h < best_h or (h == best_h and (best_v is None or v < best_v)):
                best_h = h
                best_v = v

        # If everything is unreachable (all h = inf), we return None
        if best_v is None or best_h == float("inf"):
            return None
        return best_v

    def plan_route(self, start_vertex: int) -> List[int]:
        """
        Main greedy loop:

          1. Build the initial state.
          2. While there are remaining interesting vertices:
               - choose the one with minimal heuristic (distance)
               - move there (update current, remaining, route)
          3. Return the route = order of interesting vertices to visit.

        If at some point no remaining vertex is reachable under the
        simplified model, we stop and return what we’ve built so far.
        """
        state = self._initial_state(start_vertex)
        state.route.append(start_vertex)

        # If start had people, it's already in route
        # (otherwise route starts empty)
        while state.remaining:
            next_v = self._choose_best_successor(state)
            self.expansions += 1
            if next_v is None:
                # No more legal moves in the simplified world
                break

            # "Expand" this node:
            state.route.append(next_v)
            state.remaining.remove(next_v)
            state.current = next_v

        return state.route

    
    def decide(self, obs: Observation):
        if self.action_index == 0:
            # action array logic
            self.action_array = []
            route = self.plan_route(obs.self_state.current_vertex)
            has_kit = obs.self_state.equipped
            for i in range(len(route)-1):
                self.action_array += self.base_graph.shortest_exec_path(route[i], route[i+1], obs.Q, obs.U, obs.P, has_kit)[1]

            self.action_index += 1
            action = self.action_array[0]

            if action[0] == "traverse":
                action = Action(ActionType.TRAVERSE, action[1])
            elif action[0] == "equip":
                action = Action(ActionType.EQUIP, None)
            elif action[0] == "unequip":
                action = Action(ActionType.UNEQUIP, None)

            return action
        else:
            if self.action_index < len(self.action_array):
                next_action = self.action_array[self.action_index]
            else:
                next_action = ("noop", None)
            self.action_index += 1

            if next_action[0] == "traverse":
                next_action = Action(ActionType.TRAVERSE, next_action[1])
            elif next_action[0] == "equip":
                next_action = Action(ActionType.EQUIP, None)
            elif next_action[0] == "unequip":
                next_action = Action(ActionType.UNEQUIP, None)
            elif next_action[0] == "noop":
                next_action = Action(ActionType.NO_OP, None)
                
            return next_action
            

