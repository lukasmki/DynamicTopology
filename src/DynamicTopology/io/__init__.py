"""Reaction network i/o: `.jsonl` term files, in eV and Angstrom on disk.

The same units every term is held in, so nothing converts on the way in or out.
`io.units` is the table for the q-force and OpenMM boundary (nm and kJ/mol).
Importing q-force XML is fitting-side, and lives in fast-forces
(`fastforces.qforce_xml`).
"""

from .json import read_jsonl, read_jsonls, write_jsonl, write_jsonls

__all__ = ["read_jsonl", "read_jsonls", "write_jsonl", "write_jsonls"]
