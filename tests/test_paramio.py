from pathlib import Path
from DynamicTopology.io.json import write_jsonls, read_jsonl
from DynamicTopology.io.xml import read_xml


def test_read_params_xml():
    ff_path = Path("tests/ff").resolve()
    for file in ff_path.iterdir():
        if file.suffix != ".xml":
            continue
        terms = read_xml(file.resolve())


def test_read_params_jsonl():
    ff_path = Path("tests/ff").resolve()
    for file in ff_path.iterdir():
        if file.suffix != ".jsonl":
            continue
        terms = read_jsonl(file.resolve())


def test_write_params_jsonl():
    terms = [
        {
            "type": "bond",
            "atoms": {"p1": 0, "p2": 1},
            "kwargs": {"r0": 1.0, "k": 10.0, "D": 100.0},
        }
    ]
    datastr = write_jsonls(terms)
    expect = '{"type": "bond", "atoms": {"p1": 0, "p2": 1}, "kwargs": {"r0": 1.0, "k": 10.0, "D": 100.0}}'
    assert datastr == expect
