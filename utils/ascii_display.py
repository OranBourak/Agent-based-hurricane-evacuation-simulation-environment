from typing import Dict, List, Tuple
from agent_base import AgentState

def render(time:int,Q:int,U:int,P:int,
           vertices:Dict[int,Tuple[int,bool]],
           edges:List[Tuple[int,int,int,bool]],
           agents:List[AgentState],
           scores:Dict[int,int])->None:
    print("\n" + "="*70)
    print(f"Time={time} | Q={Q} U={U} P={P}")
    by_v: Dict[int, List[str]] = {} # vertex_id -> list of agent labels
    for a in agents:
    # Add "K" if equipped
        display_label = a.label + ("K" if a.equipped else "")
        by_v.setdefault(a.current_vertex, []).append(display_label)
    print("\nVertices:")
    parts=[]
    for vid in sorted(vertices):
        p, kits =vertices[vid]
        tag=f"[{vid}"
        if p > 0: tag += f"P{p}"
        if kits > 0:
            tag += "K" if kits == 1 else f"K{kits}"
        tag += "]"
        ag="".join(sorted(by_v.get(vid, [])))
        if ag: tag+=f"<{ag}>"
        parts.append(tag)
    print("  ".join(parts))
    print("\nEdges (u--v : W, F?):")
    for u,v,w,f in sorted(edges,key=lambda e:(min(e[0],e[1]),max(e[0],e[1]))):
        print(f"  {u}--{v} : W{w}, {'F' if f else 'OK'}")
    print("\nScores:")
    for a in sorted(agents,key=lambda x:x.agent_id):
        s=scores.get(a.agent_id,0)
        print(f"  {a.label}#{a.agent_id}: {s}  (rescued={a.rescued}, actions={a.actions_done})")
    print("="*70)
