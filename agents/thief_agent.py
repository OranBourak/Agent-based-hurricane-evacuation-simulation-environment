from __future__ import annotations
from agent_base import Agent, Observation, Action, ActionType
import heapq

class ThiefAgent(Agent):
    """Seeks nearest kit, equips, then moves far from others; never rescues."""
    can_rescue = False
    def __init__(self, label: str = "thief") -> None:
        self.label = label

    def _build_adj(self, obs: Observation, allow_flooded: bool):
        adj = {}
        for u, v, w, f in obs.edges:
            if allow_flooded or not f:
                adj.setdefault(u, []).append((v, w))
                adj.setdefault(v, []).append((u, w))
        for k in adj:
            adj[k].sort(key=lambda x: x[0])
        return adj

    def _dijkstra(self, start: int, adj):
        """Return distance map from start using weights."""
        pq = [(0, start)]
        dist = {start: 0}
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]: continue
            for v, w in adj.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def _nearest_kit(self, obs: Observation):
        ''' Return the vertex id of the nearest kit, or None if none exist. '''
        kits = [vid for vid, (_, k) in obs.vertices.items() if k]
        if not kits:
            return None
        start = obs.self_state.current_vertex
        adj = self._build_adj(obs, allow_flooded=False)
        pq = [(0, start)]
        dist = {start: 0}
        while pq:
            d, u = heapq.heappop(pq)
            if u in kits:
                return u
            if d != dist[u]: continue
            for v, w in adj.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return None

    def decide(self, obs: Observation) -> Action:
        me = obs.self_state
        # --- not equipped: go for nearest kit
        if not me.equipped:
            k = self._nearest_kit(obs)
            if k is None:
                return Action(ActionType.NO_OP)
            if k == me.current_vertex:
                return Action(ActionType.EQUIP)
            adj = self._build_adj(obs, allow_flooded=False)
            pq = [(0, me.current_vertex)]
            dist = {me.current_vertex: 0}
            prev = {}
            while pq:
                d, u = heapq.heappop(pq)
                if d != dist[u]: continue
                if u == k: break
                for v, w in adj.get(u, []):
                    nd = d + w
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(pq, (nd, v))
            if k not in dist:
                return Action(ActionType.NO_OP)
            cur = k
            while prev.get(cur) != me.current_vertex:
                cur = prev[cur]
            return Action(ActionType.TRAVERSE, to_vertex=cur)
        # --- equipped: move far from others
        adj = self._build_adj(obs, allow_flooded=True)
        myd = self._dijkstra(me.current_vertex, adj) 
        others = [a for a in obs.agents if a.agent_id != me.agent_id]
        others_d = {a.agent_id: self._dijkstra(a.current_vertex, adj) for a in others}
        best_v, best_score = None, (-10**9,)
        for v, _ in adj.get(me.current_vertex, [])+[(me.current_vertex,0)]:
            min_d = min(others_d[a.agent_id].get(v, 10**9) for a in others) if others else 10**9
            score = (min_d, -myd.get(v, 0))
            if score > best_score or (score == best_score and (best_v is None or v < best_v)):
                best_score, best_v = score, v
        return Action(ActionType.TRAVERSE, to_vertex=best_v) if best_v else Action(ActionType.NO_OP)
