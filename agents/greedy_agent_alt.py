from dataclasses import dataclass
from typing import List, Set, Dict
from agent_base import Agent, Observation, Action, ActionType
from environment import Environment
from graph import Graph
from utils.heuristic_func_alt import build_transformed_graph, heuristic

@dataclass
class SearchState:
    current: int
    remaining: Set[int]
    route: List[int]
    h_value: int

INF = 10**15
class GreedyAgent(Agent):

    def __init__(self, env: Environment, label: str = "greedy") -> None:
        self.label = label
        self.base_graph = env.graph
        self.transformed_graph = build_transformed_graph(env.graph, env.P)
        self.env = env
        self.action_index = 0
        self.expansions = 0

    

    def _initial_state(self, start_vertex: int) -> SearchState:
        """Create initial search state from start vertex."""
        # "Interesting" nodes = vertices that currently have people
        remaining = {
            vid for vid, v in self.transformed_graph.vertices.items()
            if v.people > 0
        }

        route: List[int] = []
        route.append(start_vertex)

        return SearchState(
            current=start_vertex,
            remaining=remaining,
            route=route,
            h_value = heuristic(self.transformed_graph, remaining)
        )
    


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
        if state.current in state.remaining:
            state.remaining.remove(state.current)
        

        # goal test: no people remaining
        while state.remaining:
            current_vertex = state.current
            best_h = INF
            best_v = None
            # expand node
            self.expansions += 1
            for neighbor_vid, edge in self.transformed_graph.adj[current_vertex]:

                points_of_interest = set(state.remaining)
                points_of_interest.add(neighbor_vid)
                h = heuristic(self.transformed_graph, points_of_interest)
                if h < best_h:
                    best_h = h
                    best_v = neighbor_vid
            
            # creating next search state and advancing to it
            next_v = best_v
            next_remaining = set(state.remaining)
            if next_v in next_remaining:
                next_remaining.remove(next_v)
            next_route = list(state.route)
            next_route.append(next_v)
            next_state = SearchState(
                next_v,
                next_remaining,
                next_route,
                best_h
                )

            state = next_state
            if next_v is None:
                # No more legal moves in the simplified world
                break

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