import json
from pathlib import Path

import pytest
from ase import units

from DynamicTopology.io.json import read_jsonl, read_jsonls, write_jsonls
from DynamicTopology.io.units import from_disk, to_disk


def test_read_params_jsonl():
    """The legacy q-force exports in `tests/data` parse.

    Read raw: they predate the term format (`D0`, not `D`) and so have no unit
    convention for `io.units` to apply.
    """
    ff_path = Path("tests/data").resolve()
    for file in ff_path.iterdir():
        if file.suffix != ".jsonl":
            continue
        read_jsonl(file.resolve(), convert=False)


def test_write_params_jsonl():
    terms = [
        {
            "type": "bond",
            "atoms": {"p1": 0, "p2": 1},
            "kwargs": {"r0": 1.0, "k": 10.0, "D": 100.0},
        }
    ]
    datastr = write_jsonls(terms, convert=False)
    expect = '{"type": "bond", "atoms": {"p1": 0, "p2": 1}, "kwargs": {"r0": 1.0, "k": 10.0, "D": 100.0}}'
    assert datastr == expect


def test_writing_converts_to_nm_and_kj_per_mol():
    """In memory Angstrom and eV, on disk nm and kJ/mol -- and back."""
    kjmol = units.kJ / units.mol  # eV
    terms = [
        {
            "type": "bond",
            "atoms": {"p1": 0, "p2": 1},
            "kwargs": {"r0": 1.0, "k": 10.0, "D": 4.0, "h": 1.0},
        }
    ]
    row = json.loads(write_jsonls(terms))["kwargs"]
    assert row["r0"] == pytest.approx(0.1)
    assert row["k"] == pytest.approx(10.0 / kjmol * 100.0)
    assert row["D"] == pytest.approx(4.0 / kjmol)
    assert row["h"] == pytest.approx(1.0 / kjmol)
    back = read_jsonls(write_jsonls(terms))[0]["kwargs"]
    assert back == pytest.approx(terms[0]["kwargs"], rel=1e-14)


def test_unconverted_parameters_are_written_verbatim():
    """ACKS2, charges and couplings are stored in the units they are used in."""
    terms = [
        {
            "type": "threebody",
            "atoms": {"p1": 0, "p2": 1, "p3": 2},
            "kwargs": {
                "A": -2.0412345678901234,
                "a": 19.8,
                "ra0": 1.2,
                "rb0": 1.3,
                "t0": 3.1,
            },
        },
        {"type": "charge", "atoms": {"p0": 0}, "kwargs": {"q": -0.834}},
    ]
    assert read_jsonls(write_jsonls(terms)) == terms


def test_every_dataset_file_rewrites_stably():
    """A file read and written back changes no value that had 15 digits or fewer.

    `x / f * f` is not always `x`, so converted values are written to 15
    significant digits (`io.units.WRITE_DIGITS`).  Anything already stored at
    that precision -- every q-force parameter -- comes back byte-identical, and
    a second rewrite of anything is.  Without it a refit changed the last digit
    of parameters it never touched.
    """
    files = sorted(Path("datasets").glob("**/*.jsonl"))
    assert files
    for path in files:
        disk = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        once = write_jsonls(read_jsonl(path))
        assert write_jsonls(read_jsonls(once)) == once, path
        for before, after in zip(disk, (json.loads(line) for line in once.split("\n"))):
            for name, value in before["kwargs"].items():
                if len(repr(value).lstrip("-").replace(".", "").lstrip("0")) <= 15:
                    assert after["kwargs"][name] == value, (path, name)
                else:
                    assert after["kwargs"][name] == pytest.approx(value, rel=1e-14)


def test_an_unknown_parameter_is_refused():
    """A unit nobody recorded is not read in whichever unit it happened to be in."""
    with pytest.raises(KeyError, match="bond.D0"):
        from_disk("bond", "D0", 1.0)
    with pytest.raises(KeyError, match="mystery"):
        to_disk("mystery", "x", 1.0)
