import argparse
import os
from datetime import datetime
import uuid
from typing import Optional, TextIO

from environment import Environment, Parser
from agents.human_agent import HumanAgent
from agents.stupid_greedy_agent import StupidGreedyAgent
from agents.thief_agent import ThiefAgent
from agents.greedy_agent import GreedyAgent
from agents.astar_agent import AStarAgent
from agents.realtime_astar_agent import RealTimeAStarAgent


def build_demo_file(path:str)->None:
    demo = """#N 4
#U 1
#Q 2
#P 3
#V1 K
#V2 P1
#V3 B
#V4 P2 K
#E1 1 2 W1 F
#E2 3 4 W1
#E3 2 3 W1
#E4 1 3 W4
#E5 2 4 W5
"""
    with open(path,"w",encoding="utf-8") as f:
        f.write(demo)

def interactive_add_agents(env:Environment):
    ''' Interactively add agents to the environment.
    Args:
        env: the Environment instance to add agents to '''
    try:
        n=int(input("\nHow many agents? [default 3]: ") or "3")
    except:
        print("\n\033[91mInvalid input. Using default value of 3.\033[0m\n")
        n=3
    for i in range(n):
        t=(input(f"\nAgent #{i+1} type [stupid greedy/thief/human/greedy/astar/rt astar] (default is greedy): ") or "greedy").strip().lower()
        if t not in ("astar","greedy","thief","human","stupid greedy", "rt astar"):
            print(f"\n\033[91mInvalid agent type. Using default 'greedy'.\033[0m\n")
            t="stupid greedy"
        v=int(input(f"\nStart vertex for agent #{i+1} (Range: [1 - {len(env.graph.vertices)}] ): ") or "1")
        if v<1 or v>len(env.graph.vertices):
            print(f"\n\033[91mInvalid vertex. Using default value of 1.\033[0m\n")
            v=1
        if t=="stupid greedy": env.add_agent(StupidGreedyAgent(),v)
        elif t=="thief": env.add_agent(ThiefAgent(),v)
        elif t=="greedy": env.add_agent(GreedyAgent(env), v)
        elif t=="astar": env.add_agent(AStarAgent(env), v)
        elif t in ("rt astar"):
            try:
                L = int(input("Enter L (expansions per decision) [default 10]: ") or "10")
            except:
                print("\n\033[91mInvalid L. Using default 10.\033[0m\n")
                L = 10
            env.add_agent(RealTimeAStarAgent(env, L=L), v)
        else: env.add_agent(HumanAgent(),v)

def _open_log_file_interactive() -> Optional[TextIO]:
    path = (input("\nOutput log file (leave blank for terminal only): ") or "").strip()
    if not path:
        return None
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fp = open(path, "a", encoding="utf-8")  # append + create-if-missing
        print(f"Logging to '{path}' (append mode).")
        return fp
    except Exception as e:
        print(f"\n\033[91mCould not open log file '{path}': {e}. Continuing without file.\033[0m\n")
        return None

def _write_run_header(fp: TextIO, env: Environment) -> str:
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    run_id = uuid.uuid4().hex[:8]

    fp.write("\n" + "#" * 80 + "\n")
    fp.write("==== RUN START ====\n")
    fp.write(f"timestamp: {ts}\n")
    fp.write(f"run_id:    {run_id}\n")
    fp.write("#" * 80 + "\n\n")

    fp.write(env.describe() + "\n\n")
    fp.flush()
    return run_id

def _write_run_footer(fp: TextIO, run_id: str, status: str) -> None:
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    fp.write("\n" + "#" * 80 + "\n")
    fp.write("==== RUN END ====\n")
    fp.write(f"timestamp: {ts}\n")
    fp.write(f"run_id:    {run_id}\n")
    fp.write(f"status:    {status}\n")
    fp.write("#" * 80 + "\n\n")
    fp.flush()

def main():
    ap=argparse.ArgumentParser(description="Hurricane Evacuation Simulator")
    ap.add_argument("--file",type=str,default="demo_input.txt")
    ap.add_argument("--demo",action="store_true")
    ap.add_argument("--no-interactive",action="store_true")
    args=ap.parse_args()

    if args.demo:
        build_demo_file(args.file)
        print(f"Wrote demo to {args.file}")

    graph, Q, U, P = Parser.parse(args.file)

    print("\nGlobal constants from file:")
    print(f"Q={Q} (equip time), U={U} (unequip time), P={P} (speed penalty)\n")

    t = float(input("Enter T (time per expansion) [default 0]: ") or "0")

    try:
        use_custom = input("Would you like to change these values? [y/n]: ").strip().lower()
        if use_custom == "y":
            Q = int(input(f"Enter Q (equip time) [{Q}]: ") or Q)
            U = int(input(f"Enter U (unequip time) [{U}]: ") or U)
            P = int(input(f"Enter P (speed penalty) [{P}]: ") or P)
        else:
            print("Using constants from file.")
    except Exception:
        print("\033[91mInvalid input, using defaults from file.\033[0m")

    env = Environment(graph, Q=Q, U=U, P=P, T=t)

    if args.no_interactive:
        env.add_agent(StupidGreedyAgent(),1)
        env.add_agent(ThiefAgent(),4)
        env.add_agent(HumanAgent(),2)
    else:
        interactive_add_agents(env)

    log_fp = _open_log_file_interactive()
    run_id = "unknown"
    status = "finished"

    try:
        if log_fp is not None:
            run_id = _write_run_header(log_fp, env)

        print("Starting simulation...")
        env.run(max_steps=200, visualize=True, log_file=log_fp)

        if getattr(env, "simulation_done", False):
            status = "terminated"
        else:
            status = "finished"

    except Exception:
        status = "crashed"
        raise
    finally:
        if log_fp is not None:
            _write_run_footer(log_fp, run_id, status)
            log_fp.close()

if __name__=="__main__":
    main()
