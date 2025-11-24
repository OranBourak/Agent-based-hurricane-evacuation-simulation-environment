# heuristics.py
from graph import Graph, Vertex, Edge
import heapq

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


def heuristic(location, destination, graph):
    """
    Heuristic value = shortest-path distance from `location` to `destination`
    on the *transformed* graph (weights already include P for flooded edges).

    `graph` here is the transformed graph that the agent owns.
    """
    if location == destination:
        return 0

    # If your Graph already has a shortest_path method, you can use that instead
    # and just return the distance part. Here’s a generic Dijkstra, using
    # graph.neighbors(u) and edge.weight.
    pq = [(0, location)]  # (dist, vertex)
    dist = {location: 0}

    while pq:
        d, u = heapq.heappop(pq)
        if u == destination:
            return d
        if d != dist[u]:
            continue

        for v, e in graph.neighbors(u):
            nd = d + e.weight  # already transformed, no flooded logic here
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    # No path
    return float("inf")
