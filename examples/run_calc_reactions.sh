#!/bin/sh
# wB97X-V/cc-pVTZ atomization energies for every frame of every reaction.
#
# -s is the number of unpaired electrons (2S), i.e. multiplicity - 1, and is
# applied to all three frames of a file -- correct, since reactant, transition
# state and product share the total system's multiplicity.  All channels are
# neutral.  Shares tmp/atoms.json with run_calc.sh so the free-atom zero is
# identical to the one the molecule templates were referenced to.
set -e
C="--atom-cache tmp/atoms.json"

uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_01.xyz -o tmp/reactions/rxn_01.xyz $C -c 0 -s 3  # mul 4, O-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_02.xyz -o tmp/reactions/rxn_02.xyz $C -c 0 -s 2  # mul 3, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_03.xyz -o tmp/reactions/rxn_03.xyz $C -c 0 -s 1  # mul 2, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_04.xyz -o tmp/reactions/rxn_04.xyz $C -c 0 -s 2  # mul 3, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_05.xyz -o tmp/reactions/rxn_05.xyz $C -c 0 -s 0  # mul 1, dissociation
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_06.xyz -o tmp/reactions/rxn_06.xyz $C -c 0 -s 2  # mul 3, dissociation
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_07.xyz -o tmp/reactions/rxn_07.xyz $C -c 0 -s 1  # mul 2, dissociation
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_08.xyz -o tmp/reactions/rxn_08.xyz $C -c 0 -s 0  # mul 1, dissociation
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_09.xyz -o tmp/reactions/rxn_09.xyz $C -c 0 -s 1  # mul 2, dissociation
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_10.xyz -o tmp/reactions/rxn_10.xyz $C -c 0 -s 2  # mul 3, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_11.xyz -o tmp/reactions/rxn_11.xyz $C -c 0 -s 2  # mul 3, O-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_12.xyz -o tmp/reactions/rxn_12.xyz $C -c 0 -s 3  # mul 4, O-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_13.xyz -o tmp/reactions/rxn_13.xyz $C -c 0 -s 2  # mul 3, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_14.xyz -o tmp/reactions/rxn_14.xyz $C -c 0 -s 2  # mul 3, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_15.xyz -o tmp/reactions/rxn_15.xyz $C -c 0 -s 0  # mul 1, dissociation
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_16.xyz -o tmp/reactions/rxn_16.xyz $C -c 0 -s 1  # mul 2, substitution
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_17.xyz -o tmp/reactions/rxn_17.xyz $C -c 0 -s 1  # mul 2, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_18.xyz -o tmp/reactions/rxn_18.xyz $C -c 0 -s 2  # mul 3, H-transfer
uv run scripts/compute.py -i datasets/HCombustion/reactions/rxn_19.xyz -o tmp/reactions/rxn_19.xyz $C -c 0 -s 1  # mul 2, H-transfer
