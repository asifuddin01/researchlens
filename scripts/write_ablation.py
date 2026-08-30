"""Write a generated ablation table into the README, between markers.

Exists so that `make ablation` closes the loop. If the table were pasted by
hand it would drift from the code within a week, and a published number that
does not match the repository is worse than no published number.
"""

from __future__ import annotations

import sys
from pathlib import Path

START = "<!-- ablation:start -->"
END = "<!-- ablation:end -->"


def main() -> None:
    table_path, readme_path = Path(sys.argv[1]), Path(sys.argv[2])
    table = table_path.read_text().strip()
    readme = readme_path.read_text()

    if START not in readme or END not in readme:
        sys.exit(f"{readme_path} is missing the {START} / {END} markers.")

    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    readme_path.write_text(f"{head}{START}\n\n{table}\n\n{END}{tail}")


if __name__ == "__main__":
    main()
