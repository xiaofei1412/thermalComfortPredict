import argparse
import json
from pathlib import Path

from .data import ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description="Rebuild data and run prediction baselines.")
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data, audit = prepare(config, args.output / "data")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.command == "run":
        from .baseline import run
        run(data, audit, config, args.output)


if __name__ == "__main__":
    main()
