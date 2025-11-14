from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import heapq

from agent_base import Observation, Action, ActionType


def plan_segment_actions(
    obs: Observation,
    start_vertex: int,
    start_equipped: bool,
    dest_vertex: int,
) -> List[Action]:
    """
    Compute the *time-optimal* sequence of actions to go from
      (start_vertex, start_equipped)
    to
      dest_vertex (equipped or not, whichever is cheaper),
    under the real environment rules:

      - you can EQUIP only where kits > 0, cost = Q
      - you can UNEQUIP anywhere, cost = U
      - you can traverse:
          * non-flooded edges always
          * flooded edges only when equipped
        with cost = weight * (P if equipped else 1)

    Returns:
        List[Action] (possibly empty if dest unreachable).
    """
    if start_vertex == dest_vertex:
        return []

    # ---- build adjacency: u -> [(v, w, flooded), ...] ----
    adj: Dict[int, List[Tuple[int, int, bool]]] = {}
    for u, v, w, flooded in obs.edges:
        adj.setdefault(u, []).append((v, w, flooded))
        adj.setdefault(v, []).append((u, w, flooded))

    # deterministic neighbor order helps with tie-breaking
    for u in adj:
        adj[u].sort(key=lambda x: x[0])

    # ---- Dijkstra over states (vertex, equipped) ----
    State = Tuple[int, bool]

    start_state: State = (start_vertex, start_equipped)
    dist: Dict[State, int] = {start_state: 0}
    prev: Dict[State, Tuple[State, Action]] = {}
    pq: List[Tuple[int, State]] = [(0, start_state)]

    goal_state: Optional[State] = None

    while pq:
        d, state = heapq.heappop(pq)
        if d != dist[state]:
            continue

        v, equipped = state

        # If we've reached the destination vertex (in ANY equipment state),
        # we can stop – this is the optimal arrival state by Dijkstra.
        if v == dest_vertex:
            goal_state = state
            break

        people, kits = obs.vertices[v]

        # --- EQUIP (if possible) ---
        if not equipped and kits > 0:
            ns: State = (v, True)
            nd = d + obs.Q
            act = Action(ActionType.EQUIP)
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                prev[ns] = (state, act)
                heapq.heappush(pq, (nd, ns))

        # --- UNEQUIP (if currently equipped) ---
        if equipped:
            ns = (v, False)
            nd = d + obs.U
            act = Action(ActionType.UNEQUIP)
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                prev[ns] = (state, act)
                heapq.heappush(pq, (nd, ns))

        # --- TRAVERSE edges ---
        for nbr, w, flooded in adj.get(v, []):
            # can't cross flooded without kit
            if flooded and not equipped:
                continue
            cost = w * (obs.P if equipped else 1)
            ns = (nbr, equipped)
            nd = d + cost
            act = Action(ActionType.TRAVERSE, to_vertex=nbr)
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                prev[ns] = (state, act)
                heapq.heappush(pq, (nd, ns))

    # ---- reconstruct action list ----
    if goal_state is None:
        # unreachable: caller can decide what to do (e.g. TERMINATE)
        return []

    actions: List[Action] = []
    cur = goal_state
    while cur != start_state:
        prev_state, act = prev[cur]
        actions.append(act)
        cur = prev_state
    actions.reverse()
    return actions








from planning import plan_segment_actions

class SmartGreedyAgent(Agent):
    can_rescue = True

    def __init__(self, label="G2"):
        self.label = label
        self.high_route = None
        self.segment_actions: List[Action] = []
        self.segment_pos = 0
        self.route_index = 0

    def decide(self, obs: Observation) -> Action:
        # if no more planned segments, terminate
        if self.high_route is None:
            # compute high-level route ONCE using your heuristic / transformed graph
            self.high_route = compute_high_level_route_somehow(obs)
            self.route_index = 0
            self.segment_actions = []
            self.segment_pos = 0

        # if we're in the middle of executing a segment, continue it
        if self.segment_pos < len(self.segment_actions):
            act = self.segment_actions[self.segment_pos]
            self.segment_pos += 1
            return act

        # else we need to plan the next segment between route[i] and route[i+1]
        if self.route_index + 1 >= len(self.high_route):
            return Action(ActionType.TERMINATE)

        cur_v = obs.self_state.current_vertex
        dest_v = self.high_route[self.route_index + 1]
        self.segment_actions = plan_segment_actions(
            obs,
            start_vertex=cur_v,
            start_equipped=obs.self_state.equipped,
            dest_vertex=dest_v,
        )
        self.segment_pos = 0
        self.route_index += 1

        if not self.segment_actions:
            # can't reach next high-level vertex under real dynamics
            return Action(ActionType.TERMINATE)

        act = self.segment_actions[self.segment_pos]
        self.segment_pos += 1
        return act