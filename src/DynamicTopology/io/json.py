"""`.jsonl` term files: one term per line, parameters in eV and Angstrom.

The units a term is held in, so rows are read and written as they are; see
`io/units.py` for the q-force and OpenMM boundary, which is the one place a
conversion happens.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import TYPE_CHECKING

if (
    TYPE_CHECKING
):  # `core.reactionset` imports this module; a runtime import would cycle
    from DynamicTopology.core.types import Term

# No bond is this short in Angstrom (H2 is 0.74), and every bond is shorter in
# nm (0.07-0.3), so a bond `r0` under it is a file from before the `.jsonl` moved
# to eV/Angstrom.  Read as it stands it would put every length out by ten and
# every energy by 96.5, and nothing downstream would notice.
LEGACY_R0: float = 0.5


def read_jsonl(path: str | Path) -> list[Term]:
    if isinstance(path, str):
        path = Path(path).resolve()
    with open(path) as fp:
        try:
            return read_jsonls(fp.read())
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None


def write_jsonl(path: str | Path, data: list[Term], exist_ok=False) -> None:
    if isinstance(path, str):
        path = Path(path).resolve()
    if path.exists() and not exist_ok:
        raise FileExistsError(f"File `{path}` already exists.")
    with open(path, "w") as fp:
        fp.write(write_jsonls(data))


def read_jsonls(data: str) -> list[Term]:
    terms: list[Term] = []
    for line in data.split("\n"):
        if line.strip():
            term: Term = json.loads(line)
            if term["type"] == "bond" and 0 < term["kwargs"].get("r0", 1) < LEGACY_R0:
                raise ValueError(
                    f"bond r0 = {term['kwargs']['r0']} is in nm: term files are "
                    "in eV and Angstrom; convert a legacy nm/kJ/mol row with "
                    "`DynamicTopology.io.units.term_from_openmm`"
                )
            terms.append(term)
    return terms


def write_jsonls(data: list[Term]) -> str:
    return "\n".join([json.dumps(term) for term in data])
