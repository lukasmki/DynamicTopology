#!/bin/sh

uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_01.xyz -o tmp/molecules/mol_01.xyz --atom-cache tmp/atoms.json -c 0 -s 0
uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_02.xyz -o tmp/molecules/mol_02.xyz --atom-cache tmp/atoms.json -c 0 -s 2
uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_03.xyz -o tmp/molecules/mol_03.xyz --atom-cache tmp/atoms.json -c 0 -s 1
uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_04.xyz -o tmp/molecules/mol_04.xyz --atom-cache tmp/atoms.json -c 0 -s 0
uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_05.xyz -o tmp/molecules/mol_05.xyz --atom-cache tmp/atoms.json -c 0 -s 1
uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_06.xyz -o tmp/molecules/mol_06.xyz --atom-cache tmp/atoms.json -c 0 -s 0
uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_07.xyz -o tmp/molecules/mol_07.xyz --atom-cache tmp/atoms.json -c 0 -s 2
uv run scripts/compute.py -i datasets/HCombustion/molecules/mol_08.xyz -o tmp/molecules/mol_08.xyz --atom-cache tmp/atoms.json -c 0 -s 1
