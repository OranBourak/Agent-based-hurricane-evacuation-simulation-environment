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
        """Return (distance, path) using weights; optionally disallow flooded edges."""
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
