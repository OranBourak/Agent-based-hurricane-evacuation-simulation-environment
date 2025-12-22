
from typing import Dict, List, Tuple
from agent_base import AgentState

def render(time:int,Q:int,U:int,P:int,
           vertices:Dict[int,Tuple[int,bool]],
           edges:List[Tuple[int,int,int,bool]],
           agents:List[Tuple[AgentState,int]],
           scores:Dict[int,int]
           ) -> str:
    """
    Prints the state snapshot to the terminal (as before),
    AND returns the same snapshot as a single string (so we can also append it to a file).
    """
    lines: List[str] = []
    lines.append("\n" + "="*70)
    lines.append(f"Time={time} | Q={Q} U={U} P={P}")

    by_v: Dict[int, List[str]] = {} # vertex_id -> list of agent labels
    for a, _ in agents:
        display_label = f"{a.label}, K" if a.equipped else a.label
        by_v.setdefault(a.current_vertex, []).append(display_label)

    lines.append("\nVertices:")
    parts=[]
    for vid in sorted(vertices):
        p, kits = vertices[vid]
        tag=f"[{vid}"
        if p > 0: tag += f"P{p}"
        if kits > 0:
            tag += "K" if kits == 1 else f"K{kits}"
        tag += "]"
        agents_here = sorted(by_v.get(vid, []))
        for a_label in agents_here:
            tag += f"<{a_label}>"
        parts.append(tag)
    lines.append("  ".join(parts))

    lines.append("\nEdges (u--v : W, F?):")
    for u,v,w,f in sorted(edges,key=lambda e:(min(e[0],e[1]),max(e[0],e[1]))):
        lines.append(f"  {u}--{v} : W{w}, {'F' if f else 'OK'}")

    lines.append("\nScores:")
    for a, expansions in sorted(agents, key=lambda x: x[0].agent_id):
        s = scores.get(a.agent_id, 0)
        lines.append(f"  {a.label}#{a.agent_id}: {s}  (rescued={a.rescued}, actions={a.actions_done}, expansions={expansions})")

    lines.append("="*70)

    out = "\n".join(lines)
    print(out)
    return out
