from pathlib import Path
import json
from DynamicTopology.core.types import Term


def read_jsonl(path: str | Path) -> list[Term]:
    if isinstance(path, str):
        path = Path(path).resolve()
    with open(path) as fp:
        return read_jsonls(fp.read())


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
        if line:
            term: Term = json.loads(line)
            terms.append(term)
    return terms


def write_jsonls(data: list[Term]) -> str:
    return "\n".join([json.dumps(term) for term in data])
