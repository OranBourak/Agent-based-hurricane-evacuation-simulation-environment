from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import heapq


@dataclass
class Vertex:
    ''' Graph vertex representation '''
    vid: int
    people: int = 0
    kits: int = 0

@dataclass
class Edge:
    ''' Graph edge representation '''
    u: int
    v: int
    weight: int
    flooded: bool = False

@dataclass(frozen=True)
class ExecState:
    ''' Execution state for pathfinding with equipment status (vertex id and whether kit is equipped) 
    Used in shortest_exec_path function, Which finds shortest path considering equipment actions. '''
    vid: int          # vertex id
    has_kit: bool     # True / False



class Graph:
    """Undirected weighted graph; deterministic order for tie-breaking."""
    def __init__(self) -> None:
        self.vertices: Dict[int, Vertex] = {}
        self.adj: Dict[int, List[Tuple[int, Edge]]] = {} # vid -> list of (neighbor_vid, Edge)

    def add_vertex(self, vid: int, people: int = 0, kits: int = 0) -> None:
        if vid not in self.vertices: # Check if vertex already exists
            self.vertices[vid] = Vertex(vid, people, kits)
            self.adj[vid] = []
        else: # Update existing vertex
            v = self.vertices[vid]
            v.people += people
            v.kits += kits

    def add_edge(self, u: int, v: int, weight: int, flooded: bool = False) -> None:
        e = Edge(u, v, weight, flooded)
        self.adj.setdefault(u, []).append((v, e)) # Add edge to adjacency list of u, setdefault in case u not present
        self.adj.setdefault(v, []).append((u, e)) # Add edge to adjacency list of v, undirected graph
        self.adj[u].sort(key=lambda x: x[0]) # Keep neighbors sorted for deterministic order, by neighbor vid, helps in tie-breaking
        self.adj[v].sort(key=lambda x: x[0])

    def neighbors(self, vid: int) -> Iterable[Tuple[int, Edge]]:
        return self.adj.get(vid, [])

    def get_edge(self, u: int, v: int) -> Optional[Edge]:
        for w, e in self.adj.get(u, []):
            if w == v:
                return e
        return None

    def dijkstra_shortest_path(self, start: int, goal: int, allow_flooded: bool) -> Optional[Tuple[int, List[int]]]:
        """Use Dijkstra to find shortest path from start to goal, Without flooded edges if specified.
        Used in StupidGreedyAgent.
        Return (distance, path) using weights; optionally disallow flooded edges."""
        pq: List[Tuple[int, int]] = [(0, start)]
        dist: Dict[int, int] = {start: 0}
        prev: Dict[int, int] = {}
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                path = [u]
                while u in prev:
                    u = prev[u]
                    path.append(u)
                path.reverse()
                return d, path
            if d != dist[u]:
                continue
            for v, e in self.neighbors(u):
                if not allow_flooded and e.flooded:
                    continue
                nd = d + e.weight
                if v not in dist or nd < dist[v] or (nd == dist[v] and v < prev.get(v, 10**9)):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return None
    
    def compute_edge_cost(self, edge: Edge, has_kit: bool, kit_penalty_factor: int) -> float:
        """
        Compute the movement cost for this edge, given whether the agent has a kit.
        Returns:
            float("inf") if the move is illegal (flooded without kit).

        edge.flooded:
          - if has_kit: cost = weight * kit_penalty_factor
          - else:      illegal -> inf

        edge not flooded:
          - if has_kit: cost = weight * kit_penalty_factor  (penalty for carrying kit)
          - else:       cost = weight
        """

        if edge.flooded:
            if has_kit:
                return edge.weight * kit_penalty_factor
            else:
                return float("inf")

        # not flooded
        if has_kit:
            return edge.weight * kit_penalty_factor

        return edge.weight
    

    def _reconstruct_actions(
            self,
            final_state: ExecState,
            parent: Dict[ExecState, Optional[ExecState]],
            parent_action: Dict[ExecState, Optional[Tuple[str, int]]],
        ) -> List[Tuple[str, int]]:
            """
            Reconstruct a list of actions from the start state to the final_state.

            Each action is a tuple: ("equip"/"unequip"/"move", vertex_id_or_target_vid)

            Returns:
                actions: list from first action to last action.
            """
            actions_rev: List[Tuple[str, int]] = []
            cur = final_state

            # Walk back until the start_state (whose parent_action is None)
            while cur in parent_action and parent_action[cur] is not None:
                act = parent_action[cur]
                actions_rev.append(act)
                cur = parent[cur]

            actions_rev.reverse()
            return actions_rev



    def shortest_exec_path(
            self,
            start_vid: int,
            goal_vid: int,
            equip_time: int,
            unequip_time: int,
            kit_penalty_factor: int,
        ) -> Tuple[float, List[Tuple[str, int]]]:
            """
            Compute the true shortest *execution* path from start_vid to goal_vid,
            considering:
            - flooded vs unflooded edges,
            - whether the agent has an amphibian kit,
            - equip and unequip actions with given times.

            The search space is over ExecState(vid, has_kit).
            From each state we consider:
            - EQUIP (if kits > 0 at that vertex and we don't already have one),
            - UNEQUIP (if we have a kit),
            - MOVE to neighbor (if legal under flooding; cost depends on kit).

            Args:
                start_vid: starting vertex id
                goal_vid: goal vertex id
                equip_time: time cost to equip a kit
                unequip_time: time cost to unequip a kit
                kit_penalty_factor: multiplicative penalty for moving with a kit

            Returns:
                (best_cost, actions), where:
                - best_cost is the minimal total time
                - actions is a list of ("equip"/"unequip"/"move", vid/target_vid)
                    in the order they must be executed.

                If no path exists, returns (float("inf"), []).
            """

            start_state = ExecState(start_vid, False)

            # Priority queue over (cost, ExecState)
            pq: List[Tuple[float, ExecState]] = [(0.0, start_state)]
            dist: Dict[ExecState, float] = {start_state: 0.0}
            parent: Dict[ExecState, Optional[ExecState]] = {start_state: None}
            parent_action: Dict[ExecState, Optional[Tuple[str, int]]] = {start_state: None}

            while pq:
                d, state = heapq.heappop(pq)
                if d != dist.get(state, float("inf")):
                    continue

                u = state.vid

                # Goal reached (we don't care whether we have a kit at the end)
                if u == goal_vid:
                    actions = self._reconstruct_actions(state, parent, parent_action)
                    return d, actions

                # --------- EQUIP action ---------
                # Allowed only if we don't already have a kit AND there is at least one kit at u.
                if not state.has_kit and self.vertices[u].kits > 0:
                    ns = ExecState(u, True)
                    nd = d + equip_time
                    if nd < dist.get(ns, float("inf")):
                        dist[ns] = nd
                        parent[ns] = state
                        parent_action[ns] = ("equip", u)
                        heapq.heappush(pq, (nd, ns))

                # --------- UNEQUIP action ---------
                if state.has_kit:
                    ns = ExecState(u, False)
                    nd = d + unequip_time
                    if nd < dist.get(ns, float("inf")):
                        dist[ns] = nd
                        parent[ns] = state
                        parent_action[ns] = ("unequip", u)
                        heapq.heappush(pq, (nd, ns))

                # --------- MOVE actions along each edge ---------
                for v, edge in self.neighbors(u):
                    move_cost = self.compute_edge_cost(edge, state.has_kit, kit_penalty_factor)
                    if move_cost == float("inf"):
                        continue  # illegal move (e.g., flooded without kit)

                    ns = ExecState(v, state.has_kit)
                    nd = d + move_cost

                    if nd < dist.get(ns, float("inf")):
                        dist[ns] = nd
                        parent[ns] = state
                        parent_action[ns] = ("move", v)
                        heapq.heappush(pq, (nd, ns))

            # No path
            return float("inf"), []


