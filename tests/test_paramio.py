import json
from pathlib import Path

import pytest
from ase import units

from DynamicTopology.io.json import read_jsonl, read_jsonls, write_jsonls
from DynamicTopology.io.units import from_openmm, term_from_openmm, to_openmm


def test_read_params_jsonl():
    """The legacy q-force exports in `tests/data` parse.

    They predate the term format (`D0`, not `D`), and are read as they stand.
    """
    ff_path = Path("tests/data").resolve()
    for file in ff_path.iterdir():
        if file.suffix != ".jsonl":
            continue
        read_jsonl(file.resolve())


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


def test_rows_are_stored_as_held():
    """eV and Angstrom in memory and on disk: nothing converts either way."""
    terms = [
        {
            "type": "bond",
            "atoms": {"p1": 0, "p2": 1},
            "kwargs": {"r0": 0.9593275127677754, "k": 46.6, "D": 4.98, "h": 1.0},
        },
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
    assert json.loads(write_jsonls(terms).split("\n")[0]) == terms[0]
    assert read_jsonls(write_jsonls(terms)) == terms


def test_every_dataset_file_rewrites_byte_identical():
    """A file read and written back is the file.  With no conversion there is no
    `x / f * f` to round, so a refit cannot touch a parameter it did not fit."""
    files = sorted(Path("datasets").glob("**/*.jsonl"))
    assert files
    for path in files:
        text = path.read_text()
        assert write_jsonls(read_jsonl(path)) == text.rstrip("\n"), path


def test_every_dataset_bond_is_in_angstrom():
    """No bond in the shipped datasets is left in nm."""
    for path in sorted(Path("datasets").glob("**/*.jsonl")):
        for term in read_jsonl(path):
            if term["type"] == "bond":
                assert 0.5 < term["kwargs"]["r0"] < 3.0, path


def test_a_legacy_nm_file_is_refused(tmp_path):
    """A row from before the move to eV/Angstrom would read ten times too short."""
    row = {
        "type": "bond",
        "atoms": {"p1": 0, "p2": 1},
        "kwargs": {"r0": 0.09593, "k": 449613.167, "D": 481.0},
    }
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="legacy.jsonl.*in nm.*term_from_openmm"):
        read_jsonl(path)
    converted = term_from_openmm(row)
    assert read_jsonls(json.dumps(converted)) == [converted]


def test_the_openmm_table_converts_to_nm_and_kj_per_mol():
    """The q-force/OpenMM boundary: Angstrom and eV one side, nm and kJ/mol the other."""
    kjmol = units.kJ / units.mol  # eV
    assert to_openmm("bond", "r0", 1.0) == pytest.approx(0.1)
    assert to_openmm("bond", "k", 10.0) == pytest.approx(10.0 / kjmol * 100.0)
    assert to_openmm("bond", "D", 4.0) == pytest.approx(4.0 / kjmol)
    assert from_openmm("bond", "r0", 0.11433) == pytest.approx(1.1433)
    # q-force's 12-6 A/B, which only an XML import carries.
    assert from_openmm("lennardjones", "B", 1.0) == pytest.approx(1e6 * kjmol)
    # ACKS2, charges and couplings are the same number on both sides.
    assert to_openmm("threebody", "A", -2.04) == -2.04
    assert to_openmm("charge", "q", -0.834) == -0.834


def test_an_unknown_parameter_is_refused():
    """A unit nobody recorded is not converted by whichever factor happens to fit."""
    with pytest.raises(KeyError, match="bond.D0"):
        from_openmm("bond", "D0", 1.0)
    with pytest.raises(KeyError, match="mystery"):
        to_openmm("mystery", "x", 1.0)
