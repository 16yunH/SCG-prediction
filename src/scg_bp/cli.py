from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "prepare-data": "data.prepare",
    "prepare_data": "data.prepare",
    "make-splits": "splits.standard",
    "make_splits": "splits.standard",
    "make-calibrated-splits": "splits.calibrated",
    "make_calibrated_splits": "splits.calibrated",
    "train": "modeling.train",
    "baselines": "modeling.baselines",
    "advanced-baselines": "modeling.advanced_baselines",
    "advanced_baselines": "modeling.advanced_baselines",
    "evaluate": "reporting.evaluate",
    "report": "reporting.report",
    "protocol-report": "reporting.protocol",
    "protocol_report": "reporting.protocol",
}


def _usage() -> str:
    names = [
        "prepare-data",
        "make-splits",
        "make-calibrated-splits",
        "train",
        "baselines",
        "advanced-baselines",
        "evaluate",
        "report",
        "protocol-report",
    ]
    return "Usage: python -m src <command> [args]\n\nCommands:\n  " + "\n  ".join(names)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return
    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        raise SystemExit(f"Unknown command '{command}'.\n\n{_usage()}")
    sys.argv = [f"python -m src {command}", *sys.argv[2:]]
    try:
        module = importlib.import_module(f"{__package__}.{module_name}")
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Command '{command}' requires missing dependency '{e.name}'. "
            "Activate the project environment and retry."
        ) from e
    module.main()


if __name__ == "__main__":
    main()
