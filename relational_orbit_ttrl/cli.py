import argparse, json
from .simulator import REGIMES, evaluate


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the CPU controlled Relational-Orbit pilot")
    p.add_argument("--orbits", type=int, default=200)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    print(json.dumps({r.name: evaluate(r, orbits=args.orbits, k=args.rollouts, seed=args.seed)
                      for r in REGIMES}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
