from graph import Graph, Vertex, Edge
import heapq
from typing import List, Set, Dict

INF = 10**15

def build_transformed_graph(base_graph, P):
    """
    Build a copy of base_graph where:
      - flooded edges have weight = original_weight * P, Which is a penalty factor for flooded edges (P > 1)
      - flooded flag is irrelevant afterwards (we can just mark them as not flooded)
      - kits are irrelevant 

    The transformed graph is used for heuristic calculations.
    This is done ONCE per agent.
    """
    g2 = Graph()

    # Copy vertices (people still matter, kits don't)
    for vid, v in base_graph.vertices.items():
        g2.add_vertex(vid, people=v.people, kits=0)

    # Copy edges with transformed weights
    seen = set()
    for u in base_graph.vertices:
        for v, e in base_graph.neighbors(u):
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)

            new_w = e.weight * (P if e.flooded else 1)
            # Mark flooded=False because we've already 'baked in the penalty.
            g2.add_edge(u, v, new_w, flooded=False)

    return g2

def _shortest_distance_on_graph(g: Graph, u: int, v: int) -> int:
    """Return shortest-path distance between u and v on a graph."""
    if u == v:
        return 0

    res = g.dijkstra_shortest_path(u, v, allow_flooded=True)
    if res is None:
        return INF

    dist, _path = res
    return int(dist)


def heuristic(g: Graph, vertices: List[int]) -> int:
    """recieves a transformed graph and a list of nodes of interest
       returns the weight of the minimum spanning tree covering those nodes in the graph"""
    vertices = sorted(set(vertices))
    n = len(vertices)

    if n <= 1:
        return 0

    # calculating shortest distance between each pair of vertices
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _shortest_distance_on_graph(g, vertices[i], vertices[j])
            dist[i][j] = d
            dist[j][i] = d

    # running prim's algorithm to calculate MST
    in_mst = [False] * n
    min_edge = [INF] * n
    min_edge[0] = 0

    total_cost = 0 # sum of edge weights in the tree

    for _ in range(n):
        u = -1
        best = INF

        # pick best vertex to add
        for i in range(n):
            if not in_mst[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i

        if u == -1 or best == INF:
            return INF   # not all nodes are reachable

        in_mst[u] = True
        total_cost += min_edge[u]

        # relax neighbors
        for v in range(n):
            if not in_mst[v] and dist[u][v] < min_edge[v]:
                min_edge[v] = dist[u][v]

    return total_cost
