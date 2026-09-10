#!/usr/bin/env python3
"""Check that notebooks have no error outputs and ran top to bottom in order.

Static check only, does not execute the notebook.
"""

import json
import sys


def check_notebook(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    errors = []
    expected_count = 1
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"cell {i}: has an error output ({output.get('ename')})")

        count = cell.get("execution_count")
        if count is None:
            errors.append(f"cell {i}: was never executed")
            continue
        if count != expected_count:
            errors.append(
                f"cell {i}: executed out of order (expected {expected_count}, got {count})"
            )
        expected_count = count + 1

    return errors


def main(paths: list[str]) -> int:
    ok = True
    for path in paths:
        for error in check_notebook(path):
            print(f"{path}: {error}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
