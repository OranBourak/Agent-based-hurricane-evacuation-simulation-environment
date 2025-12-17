from typing import Optional
from agent_base import Agent, Observation, Action, ActionType
from environment import Environment
from utils.heuristic_func import build_transformed_graph, heuristic


class GreedyAgent(Agent):
    """
    ONLINE Greedy Best-First agent.
    One expansion for one action.
    Priority = heuristic only.
    """

    def __init__(self, env: Environment, label: str = "greedy") -> None:
        self.label = label
        self.base_graph = env.graph
        self.transformed_graph = build_transformed_graph(env.graph, env.P)
        self.env = env
        self.expansions = 0

    # ------------------------------------------------------------
    # DECISION (ONE EXPANSION PER CALL)
    # ------------------------------------------------------------
    def decide(self, obs: Observation) -> Action:
        self.expansions += 1  # expansion

        u = obs.self_state.current_vertex
        equipped = obs.self_state.equipped
        kits_here = obs.vertices[u][1]

        # remaining people (for heuristic)
        remaining = {
            vid for vid, (ppl, _) in obs.vertices.items()
            if ppl > 0 and vid != u
        }

        best_action: Optional[Action] = None
        best_h = float("inf")

        # ------------------------------------------------
        # EQUIP 
        # ------------------------------------------------
        if not equipped and kits_here > 0:
            h = heuristic(self.transformed_graph, [u] + list(remaining))
            if h < best_h:
                best_h = h
                best_action = Action(ActionType.EQUIP)

        # ------------------------------------------------
        # UNEQUIP 
        # ------------------------------------------------
        if equipped:
            h = heuristic(self.transformed_graph, [u] + list(remaining))
            if h < best_h:
                best_h = h
                best_action = Action(ActionType.UNEQUIP)

        # ------------------------------------------------
        # TRAVERSE (legal only)
        # ------------------------------------------------
        for v, edge in self.base_graph.neighbors(u):
            if edge.flooded and not equipped:
                continue  # illegal

            new_remaining = set(remaining)
            new_remaining.discard(v)

            h = heuristic(
                self.transformed_graph,
                [v] + list(new_remaining)
            )

            if h < best_h:
                best_h = h
                best_action = Action(ActionType.TRAVERSE, v)

        # ------------------------------------------------
        # No legal action
        # ------------------------------------------------
        if best_action is None:
            return Action(ActionType.NO_OP)

        return best_action
