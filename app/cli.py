"""Command-line entry point: `python -m app.cli --help`.

`hello` checks the wiring; every other command (split, train, predict, generate, write,
evaluate, demo) is handled by main.py.
"""
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["hello"]:
        print("Creative Writing Prompter is ready. Open README.md for your milestones.")
        return 0
    from main import main as wordflow_main

    return wordflow_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
