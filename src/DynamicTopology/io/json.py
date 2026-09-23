"""`.jsonl` term files: one term per line, parameters in nm and kJ/mol.

Reading converts to the ASE units every term is held in, and writing converts
back; see `io/units.py`.  `convert=False` moves rows through untouched, for a
caller whose terms are already in the on-disk units (a q-force XML import).
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import TYPE_CHECKING

from DynamicTopology.io.units import term_from_disk, term_to_disk

if (
    TYPE_CHECKING
):  # `core.reactionset` imports this module; a runtime import would cycle
    from DynamicTopology.core.types import Term


def read_jsonl(path: str | Path, convert: bool = True) -> list[Term]:
    if isinstance(path, str):
        path = Path(path).resolve()
    with open(path) as fp:
        return read_jsonls(fp.read(), convert=convert)


def write_jsonl(
    path: str | Path, data: list[Term], exist_ok=False, convert: bool = True
) -> None:
    if isinstance(path, str):
        path = Path(path).resolve()
    if path.exists() and not exist_ok:
        raise FileExistsError(f"File `{path}` already exists.")
    with open(path, "w") as fp:
        fp.write(write_jsonls(data, convert=convert))


def read_jsonls(data: str, convert: bool = True) -> list[Term]:
    terms: list[Term] = []
    for line in data.split("\n"):
        if line.strip():
            term: Term = json.loads(line)
            terms.append(term_from_disk(term) if convert else term)
    return terms


def write_jsonls(data: list[Term], convert: bool = True) -> str:
    return "\n".join(
        [json.dumps(term_to_disk(term) if convert else term) for term in data]
    )
