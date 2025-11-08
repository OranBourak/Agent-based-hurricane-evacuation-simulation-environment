import argparse
from environment import Environment, Parser
from agents.human_agent import HumanAgent
from agents.greedy_agent import GreedyAgent
from agents.thief_agent import ThiefAgent

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
        print("\nInvalid input. Using default value of 3.")
        n=3
    for i in range(n):
        t=(input(f"\nAgent #{i+1} type [greedy/thief/human] (default greedy): ") or "greedy").strip().lower()
        if t not in ("greedy","thief","human"): t="greedy"
        v=int(input(f"\nStart vertex for agent #{i+1} (Range: [1 - {len(env.graph.vertices)}] ): ") or "1")
        if v<1 or v>len(env.graph.vertices):
            print(f"\nInvalid vertex. Using default value of 1.\n")
            v=1
        if t=="greedy": env.add_agent(GreedyAgent(),v)
        elif t=="thief": env.add_agent(ThiefAgent(),v)
        else: env.add_agent(HumanAgent(),v)

def main():
    ap=argparse.ArgumentParser(description="Hurricane Evacuation Simulator")
    ap.add_argument("--file",type=str,default="demo_input.txt")
    ap.add_argument("--demo",action="store_true")
    ap.add_argument("--no-interactive",action="store_true")
    args=ap.parse_args()

    if args.demo: # create demo file if flagged
        build_demo_file(args.file)
        print(f"Wrote demo to {args.file}")

    graph,Q,U,P=Parser.parse(args.file)
    env=Environment(graph,Q,U,P)

    if args.no_interactive:
        env.add_agent(GreedyAgent(),1)
        env.add_agent(ThiefAgent(),4)
        env.add_agent(HumanAgent(),2)
    else:
        interactive_add_agents(env)

    print("Starting simulation...")
    env.run(max_steps=200,visualize=True)

if __name__=="__main__":
    main()
