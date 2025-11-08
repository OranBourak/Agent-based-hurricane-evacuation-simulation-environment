from __future__ import annotations
from agent_base import Agent, Observation, Action, ActionType
import heapq

class GreedyAgent(Agent):
    """Moves toward nearest people via shortest UNFLOODED path."""
    can_rescue = True
    def __init__(self, label: str = "G") -> None:
        self.label = label

    def _build_adj(self, obs: Observation):
        adj = {}
        for u, v, w, f in obs.edges:
            if f:  # flooded roads are blocked
                continue
            adj.setdefault(u, []).append((v, w)) # add edge u->v
            adj.setdefault(v, []).append((u, w)) # add edge v->u
        for k in adj:
            adj[k].sort(key=lambda x: x[0])
        return adj

    def _nearest_people(self, obs: Observation):
        targets = [vid for vid, (p, _) in obs.vertices.items() if p > 0]
        if not targets:
            return None
        start = obs.self_state.current_vertex
        adj = self._build_adj(obs)
        pq = [(0, start)] # priority queue of (distance, vertex)
        dist = {start: 0}
        while pq:
            d, u = heapq.heappop(pq) # get vertex with smallest distance, d is distance to u
            if u in targets:
                return u
            if d != dist[u]:
                continue
            for v, w in adj.get(u, []): # explore neighbors of u
                nd = d + w # new distance to neighbor v
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return None

    def _next_step(self, obs: Observation, goal: int):
        ''' Return the next vertex to step to on the shortest path to goal. '''
        adj = self._build_adj(obs)
        start = obs.self_state.current_vertex
        pq = [(0, start)]
        dist = {start: 0}
        prev = {}
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            if u == goal:
                break
            for v, w in adj.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if goal not in dist:
            return None
        cur = goal
        while prev.get(cur) != start:
            cur = prev[cur]
        return cur

    def decide(self, obs: Observation) -> Action:
        goal = self._nearest_people(obs)
        if goal is None:
            return Action(ActionType.TERMINATE)
        if goal == obs.self_state.current_vertex:
            return Action(ActionType.NO_OP)
        step = self._next_step(obs, goal)
        return Action(ActionType.TRAVERSE, to_vertex=step) if step else Action(ActionType.TERMINATE)
