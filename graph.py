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
    
    
    def compute_edge_cost(edge, has_kit):
        """
        Compute the real movement cost for this edge.
        Handles flooded/unflooded logic and whether kit is equipped.
        Returns float("inf") for illegal moves.
        """

        # Case 1: edge is flooded
        if edge.flooded:
            if has_kit:
                # You can cross it at normal cost (or maybe discounted, depending on spec)
                return edge.weight
            else:
                # Cannot cross flooded road without kit
                return float("inf")

        # Case 2: edge is NOT flooded
        if has_kit:
            # Some versions of the assignment penalize unflooded travel with a kit
            # If your rules say there's no penalty, change this to "return edge.weight"
            return edge.weight       # or penalty if assignment says so

        # Case 3: normal case: no flooded road, no kit
        return edge.weight


    def reconstruct_path(final_state, parent, parent_step):
        """ Reconstruct the physical path (list of vertex ids) from start to goal,
        given the parent mapping of ExecStates and the vertex transitions."""
        path_states = []
        cur = final_state
        while cur is not None:
            path_states.append(cur)
            cur = parent[cur]
        path_states.reverse()

        # extract only the vertices
        path_vertices = []
        last_vid = None
        for st in path_states:
            if st.vid != last_vid:
                path_vertices.append(st.vid)
                last_vid = st.vid

        return path_vertices



    def shortest_exec_path(
        graph,
        start_vid: int,
        goal_vid: int,
        equip_time: int,
        unequip_time: int,
    ):
        """
        True shortest path from start_vid to goal_vid,
        considering REAL costs: kit rules, flooded roads, equip/unequip actions.

        Uses Dijkstra's algorithm on an expanded state space (vid, has_kit).
        The search space includes equip/unequip actions at each vertex.
        That search space considers all possible states and transitions can be existed by:
        - Equipping a kit (if available at current vertex)
        - Unequipping a kit
        - Moving along edges (with costs depending on flooded status and kit possession)

        Returns a list of vertex ids for the optimal physical path.
        """

        start_state = ExecState(start_vid, False)

        pq = [(0, start_state)]
        dist = {start_state: 0}
        parent = {start_state: None}
        parent_step = {start_state: None}  # store vertex transitions

        while pq:
            d, state = heapq.heappop(pq)
            if d != dist[state]:
                continue

            u = state.vid

            # Goal reached (kit status doesn't matter).
            if u == goal_vid:
                return reconstruct_path(state, parent, parent_step)

            # ---------- Equip action ----------
            if not state.has_kit and graph.vertices[u].has_kit:
                ns = ExecState(u, True)
                nd = d + equip_time
                if nd < dist.get(ns, float("inf")):
                    dist[ns] = nd
                    parent[ns] = state
                    parent_step[ns] = u
                    heapq.heappush(pq, (nd, ns))

            # ---------- Unequip action ----------
            if state.has_kit:
                ns = ExecState(u, False)
                nd = d + unequip_time
                if nd < dist.get(ns, float("inf")):
                    dist[ns] = nd
                    parent[ns] = state
                    parent_step[ns] = u
                    heapq.heappush(pq, (nd, ns))

            # ---------- Move along edges ----------
            for v, edge in graph.neighbors(u):
                move_cost = compute_edge_cost(edge, state.has_kit)
                if move_cost == float("inf"):
                    continue

                ns = ExecState(v, state.has_kit)
                nd = d + move_cost

                if nd < dist.get(ns, float("inf")):
                    dist[ns] = nd
                    parent[ns] = state
                    parent_step[ns] = v
                    heapq.heappush(pq, (nd, ns))

        # No path
        return []


