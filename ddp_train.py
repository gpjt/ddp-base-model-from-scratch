import json
from pathlib import Path

import click


@click.command()
@click.argument("run")
def main(run):
    run_dir = Path(__file__).resolve().parent / "runs" / run
    if not run_dir.is_dir():
        raise Exception(f"Could not find run dir {run_dir}")

    model_conf_file = run_dir / "model.json"
    if not model_conf_file.is_file():
        raise Exception(f"Could not find model config in {model_conf_file}")
    with open(model_conf_file, "r") as f:
        model_conf = json.load(f)

    print(model_conf)


if __name__ == "__main__":
    main()
