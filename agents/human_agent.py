from __future__ import annotations
from agent_base import Agent, Observation, Action, ActionType

class HumanAgent(Agent):
    can_rescue = True
    def __init__(self, label: str = "H") -> None:
        self.label = label

    def decide(self, obs: Observation) -> Action:
        print(f"\n[HUMAN] Time={obs.time}, current vertex={obs.self_state.current_vertex}")
        print("Actions: traverse <v> | equip | unequip | noop | term")
        try:
            s = input("move> ").strip().lower()
        except EOFError:
            print("\nEOF received, doing NO_OP.")
            return Action(ActionType.NO_OP)
        parts = s.split()
        if not parts:
            return Action(ActionType.NO_OP)
        if parts[0] in ("t", "traverse") and len(parts) == 2 and parts[1].isdigit():
            return Action(ActionType.TRAVERSE, to_vertex=int(parts[1]))
        if parts[0] in ("equip", "q"): return Action(ActionType.EQUIP)
        if parts[0] in ("unequip", "u"): return Action(ActionType.UNEQUIP)
        if parts[0] in ("noop", "n"): return Action(ActionType.NO_OP)
        if parts[0] in ("term", "terminate"): return Action(ActionType.TERMINATE)
        print("Unrecognized command; doing NO_OP.")
        return Action(ActionType.NO_OP)
