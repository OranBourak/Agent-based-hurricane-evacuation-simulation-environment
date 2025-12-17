from typing import Dict, List, Tuple, Iterable
from dataclasses import dataclass
from agent_base import Agent, Observation, AgentState, Action, ActionType
from graph import Graph
from utils.ascii_display import render
from collections import deque

# ---------------- Environment ----------------

@dataclass
class EnvAgentRecord:
    agent: Agent
    state: AgentState

class Environment:
    """Turn-based simulator for hurricane evacuation agents."""
    def __init__(self, graph: Graph, Q: int, U: int, P: int, T:float) -> None:
        self.graph = graph
        self.Q, self.U, self.P = Q, U, P
        self.time = 0
        self.agents: List[EnvAgentRecord] = []
        self.simulation_done = False
        self.T = T

    # ---- registration ----
    def add_agent(self, agent: Agent, start_vertex: int, equipped: bool = False) -> int:
        aid = len(self.agents)
        st = AgentState(aid, agent.label, start_vertex, equipped, 0, 0)
        self.agents.append(EnvAgentRecord(agent, st))
        return aid

    # ---- main loop ----
    def run(self, max_steps: int = 1000, visualize: bool = True) -> None:
        ''' Run the environment simulation for a maximum number of steps.'''
        # Initial display
        if visualize: self._display()

        for _ in range(max_steps):
            if self._all_rescued() or not self._any_agent_can_reach_people(): # termination conditions
                if visualize: self._display()
                print("\n Simulation completed successfully. All people have been rescued or are unreachable.\n")
                break
            # ----  agent turns loop ----
            for rec in self.agents: # each agent's turn
                if self._all_rescued() or not self._any_agent_can_reach_people():
                    if visualize: self._display()
                    print("\n\033[91mAll people have been rescued or no agent can reach remaining people. Simulation ends.\033[0m")
                    return
                self._auto_rescue(rec) # auto-rescue before next action (required by assignment to be done right before action)
                obs = self._make_obs(rec)
                act = rec.agent.decide(obs)
                if visualize: self._display()
                self._apply_action(rec, act)
                self._auto_rescue(rec)


    # ---- helpers ----
    def _auto_rescue(self, rec: EnvAgentRecord) -> None:
        ''' Automatically rescue people at current vertex if possible. '''
        if not getattr(rec.agent, "can_rescue", True):
            return
        v = self.graph.vertices[rec.state.current_vertex]
        # if there are people here, rescue them all
        if v.people > 0:
            rec.state = AgentState(
                rec.state.agent_id,
                rec.state.label,
                rec.state.current_vertex,
                rec.state.equipped,
                rec.state.rescued + v.people,
                rec.state.actions_done,
            )
            v.people = 0

    def _make_obs(self, rec: EnvAgentRecord) -> Observation:
        ''' Create an Observation for the given agent record. '''
        verts = {vid: (v.people, v.kits) for vid, v in self.graph.vertices.items()} # vertex_id -> (people, kits)
        edges: List[Tuple[int,int,int,bool]] = []
        seen=set()
        for u in self.graph.adj.keys():
            for v,e in self.graph.adj[u]:
                key=tuple(sorted((u,v)))
                if key in seen: continue
                seen.add(key)
                edges.append((e.u,e.v,e.weight,e.flooded))
        return Observation(self.time,self.Q,self.U,self.P,verts,edges,[a.state for a in self.agents],rec.state)

    def _apply_action(self, rec: EnvAgentRecord, act: Action) -> None:
        st = rec.state
        rec.state = st

        if act.kind == ActionType.TERMINATE:
            # Check if simulation already finished naturally
            if self._all_rescued() or not self._any_agent_can_reach_people():
                self.simulation_done = True
                return  # don't exit here; let run() finish cleanly
            else:
                print("\033[91m\nSimulation terminated manually by user.\033[0m")
                exit(0)
        
        # Apply action effects and update time
        if act.kind == ActionType.NO_OP:
            st = AgentState(st.agent_id, st.label, st.current_vertex, st.equipped, st.rescued, st.actions_done + 1) 
            self.time += 1
            return
        if act.kind == ActionType.EQUIP:
            v = self.graph.vertices[st.current_vertex]
            if (not st.equipped) and v.kits > 0:
                self.time += self.Q
                v.kits -= 1
                rec.state = AgentState(st.agent_id, st.label, st.current_vertex, True, st.rescued, st.actions_done + 1)
            else: 
                self.time += 1
            return
        if act.kind == ActionType.UNEQUIP:
            if st.equipped:
                self.time += self.U
                self.graph.vertices[st.current_vertex].kits += 1
                rec.state = AgentState(st.agent_id, st.label, st.current_vertex, False, st.rescued, st.actions_done + 1)
            else:
                self.time += 1
            return
        if act.kind == ActionType.TRAVERSE:
            to_v = act.to_vertex
            if to_v is None:
                self.time += 1
                return
            e = self.graph.get_edge(st.current_vertex, to_v)
            if e is None or (e.flooded and not st.equipped):
                self.time += 1
                return
            cost = e.weight * (self.P if st.equipped else 1)
            self.time += cost
            rec.state = AgentState(st.agent_id, st.label, to_v, st.equipped, st.rescued, st.actions_done + 1)

    # ---- termination ----
    def _all_rescued(self) -> bool:
        ''' Check if all people have been rescued. 
        returns: True if all people rescued, False otherwise '''
        return all(v.people == 0 for v in self.graph.vertices.values())

    def _any_agent_can_reach_people(self) -> bool:
        ''' Check if any agent can still reach people.
        returns: True if any agent can reach people, False otherwise '''

        people_vs = {vid for vid,v in self.graph.vertices.items() if v.people>0}
        if not people_vs: return True

        for rec in self.agents:
            st = rec.state
            if st.equipped: # check if can reach people directly, when equipped
                if self._reachable_any(st.current_vertex, people_vs, True): return True
            else:
                if self._reachable_any(st.current_vertex, people_vs, False): return True  # check if can reach people directly, when not equipped
               
                # Check if the agent can reach any kit first, if so then check if from there can reach people after equipping
                kit_vs = {vid for vid,v in self.graph.vertices.items() if v.kits > 0}
                for kv in kit_vs:
                    if self._reachable(st.current_vertex, kv, False) and \
                       self._reachable_any(kv, people_vs, True):
                        return True
        return False

    def _reachable(self,s:int,t:int,allow_flooded:bool)->bool:
        ''' Check if t is reachable from s under flooding constraints. '''
        seen={s}; dq=deque([s])
        while dq:
            u=dq.popleft()
            if u==t: return True
            for v,e in self.graph.neighbors(u):
                if not allow_flooded and e.flooded: continue
                if v in seen: continue
                seen.add(v); dq.append(v)
        return False

    def _reachable_any(self,s:int,targets:Iterable[int],allow_flooded:bool)->bool:
        ''' Check if any of targets is reachable from s under flooding constraints. 
        Args:
            s: start vertex id
            targets: iterable of target vertex ids
            allow_flooded: whether flooded edges can be traversed'''
        seen={s}; dq=deque([s]); targets=set(targets)
        while dq:
            u=dq.popleft()
            if u in targets: return True
            for v,e in self.graph.neighbors(u):
                if not allow_flooded and e.flooded: continue
                if v in seen: continue
                seen.add(v); dq.append(v)
        return False

    def _display(self):
        ''' Display the current state of the environment using ASCII rendering. '''
        scores = dict()
        for agent in self.agents:
            if agent.agent.label in ["stupid greedy", "thief", "human"]:
                scores[agent.state.agent_id] = agent.state.rescued * 1000 - self.time
            else:
                scores[agent.state.agent_id] =  agent.state.rescued * 1000 - (self.time + agent.agent.expansions * self.T)
        verts = {vid:(v.people,v.kits) for vid,v in self.graph.vertices.items()}
        edges=[]
        seen=set()
        for u in self.graph.adj:
            for v,e in self.graph.adj[u]:
                key=tuple(sorted((u,v)))
                if key in seen: continue
                seen.add(key)
                edges.append((e.u,e.v,e.weight,e.flooded))
        render(self.time,self.Q,self.U,self.P,verts,edges,[a.state for a in self.agents],scores)

# ---------------- Parser ----------------
class Parser:
    """Parses the ASCII world file as described in assignment."""
    @staticmethod
    def parse(path: str) -> tuple[Graph,int,int,int]:
        g = Graph(); Q=2; U=1; P=3
        with open(path,"r",encoding="utf-8") as f:
            for raw in f:
                line=raw.strip()
                if not line or line.startswith(";"): continue
                semi=line.find(";")
                if semi!=-1: line=line[:semi].strip()
                if not line: continue
                toks=line.split(); tag=toks[0].upper()
                if tag=="#U": U=int(toks[1])
                elif tag=="#Q": Q=int(toks[1])
                elif tag=="#P": P=int(toks[1])
                elif tag.startswith("#V"):
                    vid = int(tag[2:])
                    people = kits = 0
                    for t in toks[1:]:
                        up = t.upper()
                        if up.startswith("P"):
                            people += int(up[1:])
                        elif up.startswith("K"):
                            # allow K, K2, or multiple Ks
                            kits += int(up[1:] or "1")
                    g.add_vertex(vid, people, kits)
                elif tag.startswith("#E"):
                    u=int(toks[1]); v=int(toks[2])
                    wtok=toks[3].upper(); w=int(wtok[1:])
                    flooded=any(t.upper()=="F" for t in toks[4:])
                    if u not in g.vertices: g.add_vertex(u)
                    if v not in g.vertices: g.add_vertex(v)
                    g.add_edge(u,v,w,flooded)
        return g,Q,U,P
