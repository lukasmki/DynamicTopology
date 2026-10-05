"""The 12-6 takes each diabatic state's parameters from that state's templates.

Every shipped template gives an atom its element's sigma and epsilon, so the
whole-system 12-6 is one number for every diabatic state and sits outside the
EVB Hamiltonian.  Per-molecule fitting breaks that: H3O+'s oxygen need not be
H2O's, and a state is then owed the 12-6 of *its own* parameters -- which is
also what its `exclusion` terms, built per template, already assume.

The perturbed set below is the Water dataset with H3O+ and OH- given their own
parameters.  `ReactionSet.load` re-derives the exclusions from them, so every
template is still self-consistent and only the state-dependence is new.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.evb import EVBSystem
from DynamicTopology.forcefield.evaluate import evaluate
from DynamicTopology.forcefield.lj import LennardJones
from DynamicTopology.forcefield.params import ForceFieldParams, active, use
from DynamicTopology.system import System

from geometry import REACTION, REACTION_PATH_RAMP, reaction_path
from test_stress import finite_difference_virial

WATER = Path("datasets/Water")
HCOMBUSTION = Path("datasets/HCombustion/HCombustion.json")

# `(sigma, eps)` for O and for H, in Angstrom and eV.  Far enough from the
# element values (O 3.05 / 0.00737, H 0 / 0) that every difference is well
# above round-off, and H given a radius so its pairs take part at all.
PERTURBED = {
    "h3o": ((3.15, 0.0096), (0.8, 1e-3)),
    "h1o": ((3.00, 0.0060), (0.0, 0.0)),
}

# Fraction of the way from a proton transfer's transition state back to its
# reactant: off the symmetric TS, so the two states' 12-6 sums differ, and
# still close enough that both carry weight.
OFF_TS = 0.3


def _params(manifest: Path) -> ForceFieldParams:
    return ForceFieldParams.from_dict(
        json.loads(manifest.read_text())["global_params"]
    )


@pytest.fixture(scope="module")
def perturbed(tmp_path_factory):
    """The Water dataset with per-molecule 12-6 parameters for H3O+ and OH-."""
    root = tmp_path_factory.mktemp("lj") / "Water"
    shutil.copytree(WATER, root)
    for name, (oxygen, hydrogen) in PERTURBED.items():
        path = root / "molecules" / f"{name}.jsonl"
        numbers = io.read(path.with_suffix(".xyz")).numbers
        lines = []
        for line in path.read_text().splitlines():
            term = json.loads(line)
            if term["type"] == "lennardjones":
                element = numbers[term["atoms"]["p0"]]
                sigma, eps = oxygen if element == 8 else hydrogen
                term["kwargs"] = {"sigma": sigma, "eps": eps}
            lines.append(json.dumps(term))
        path.write_text("\n".join(lines) + "\n")
    manifest = root / "Water.json"
    params = _params(manifest)
    with use(params):
        reaction_set = ReactionSet(manifest)
    return root, params, reaction_set


@pytest.fixture
def water(perturbed):
    """The perturbed set, with its parameters in force for the test."""
    root, params, reaction_set = perturbed
    with use(params):
        yield root, reaction_set


def proton_transfer(root: Path, name: str, t: float) -> tuple[Atoms, Atoms]:
    """`(geometry, reactant)`: `t` of the way from `name`'s TS to its reactant.

    The reactant frame keeps its `connectivity`, so `Topology.from_atoms` reads
    the reactant's bonds off it; the geometry has none, as a trajectory frame
    would not.  Centred on the origin.
    """
    frames = io.read(root / f"reactions/{name}.xyz", index=":")
    transition = frames[len(frames) // 2]
    reactant = frames[0].copy()
    reactant.calc = None
    reactant.positions = transition.positions + t * (
        frames[0].positions - transition.positions
    )
    reactant.positions -= reactant.positions.mean(0)
    atoms = reactant.copy()
    atoms.info.pop("connectivity", None)
    return atoms, reactant


def zundel(root: Path, t: float = OFF_TS) -> tuple[Atoms, Topology]:
    atoms, reactant = proton_transfer(root, "h3o-h2o-transfer", t)
    atoms.set_cell(np.eye(3) * 12.0)
    atoms.positions += 6.0
    atoms.pbc = False
    return atoms, Topology.from_atoms(reactant)


def two_transfers(root: Path) -> tuple[Atoms, Topology]:
    """H3O+.H2O and H2O.OH-, far enough apart to be separate blocks.

    Periodic, so the minimum image and the virial are both in it; 9 Angstrom
    between the centres keeps every intercomplex pair outside the bimolecular
    cutoff while leaving the 12-6 between the two blocks nonzero.
    """
    a, ra = proton_transfer(root, "h3o-h2o-transfer", OFF_TS)
    b, rb = proton_transfer(root, "h2o-oh-transfer", OFF_TS)
    length = 18.0
    for x in (a, ra):
        x.positions += [5.0, length / 2, length / 2]
    for x in (b, rb):
        x.positions += [13.0, length / 2, length / 2]
    atoms = a + b
    reference = ra + rb
    n = len(a)
    reference.info["connectivity"] = ra.info["connectivity"] + [
        [i + n, j + n, order] for i, j, order in rb.info["connectivity"]
    ]
    atoms.info.pop("connectivity", None)
    atoms.set_cell(np.eye(3) * length)
    atoms.pbc = True
    return atoms, Topology.from_atoms(reference)


def lj_of(atoms: Atoms, state: Topology) -> float:
    return LennardJones()(atoms.positions, atoms.pbc, atoms.cell, state.term_dict)[0]


# -- agreeing templates ------------------------------------------------------


@pytest.mark.parametrize("periodic", [False, True], ids=["open", "periodic"])
def test_agreeing_templates_change_nothing(periodic):
    """Per-element parameters: no block varies and the sum is the seed's.

    `evaluate` then takes `__call__` on the seed topology, so this is the number
    the 12-6 gave before it learned about states, to the last bit.
    """
    with use(_params(HCOMBUSTION)):
        reaction_set = ReactionSet(HCOMBUSTION)
        atoms = reaction_path(REACTION, REACTION_PATH_RAMP, 12.0)
        atoms.pbc = periodic
        system = System(atoms, Topology.from_atoms(atoms), reaction_set)
        seed = system.topology
        results = system.calculate()

        assert any(block["nstates"] > 1 for block in results["blocks"])
        assert not any(block.varies for block in system.lj_ff.blocks)
        for i, block in enumerate(system.lj_ff.blocks):
            assert np.all(system.lj_ff.corrections(i) == 0.0)
        expected = LennardJones()(atoms.positions, atoms.pbc, atoms.cell, seed.term_dict)
        assert results["energy_lj"] == expected[0]


# -- one block ---------------------------------------------------------------


def test_each_state_is_charged_its_own_sum(water):
    """With one block spanning everything, each correction *is* that state's sum.

    And the reported 12-6 is their ground-state average: the states of a block
    are alternatives, so it is the pair energies that are averaged and not the
    parameters.
    """
    root, reaction_set = water
    atoms, topology = zundel(root)
    system = System(atoms, topology, reaction_set)
    results = system.calculate()

    (block,) = system.basis.build(atoms, system.topology, system.bimol_cutoff)
    assert block.nstates > 1
    lj = system.lj_ff
    assert lj.blocks[0].varies

    brute = np.array([lj_of(atoms, state) for state in block.states])
    assert np.ptp(brute) > 1e-6, "the states' 12-6 sums agree; the test is vacuous"
    np.testing.assert_allclose(lj.corrections(0), brute, rtol=0, atol=1e-12)
    weights = results["blocks"][0]["weights"]
    assert results["energy_lj"] == pytest.approx(weights @ brute, abs=1e-12)


def test_the_answer_does_not_depend_on_the_seed(water):
    """Seeded from either end of the transfer, the same surface.

    This is what failed before: the whole-system sum took the *seed's*
    parameters, so starting from H3O+ on the other oxygen charged every state
    the other molecule's 12-6.
    """
    root, reaction_set = water
    atoms, reactant = zundel(root)
    frames = io.read(root / "reactions/h3o-h2o-transfer.xyz", index=":")
    product = frames[-1].copy()
    product.calc = None
    a = System(atoms, reactant, reaction_set).calculate()
    b = System(atoms, Topology.from_atoms(product), reaction_set).calculate()
    assert a["energy_lj"] == pytest.approx(b["energy_lj"], abs=1e-10)
    assert a["energy"] == pytest.approx(b["energy"], abs=1e-10)
    np.testing.assert_allclose(a["forces"], b["forces"], atol=1e-9)


@pytest.mark.parametrize("periodic", [False, True], ids=["open", "periodic"])
def test_a_lone_template_scores_as_system_scores_it(water, periodic):
    """The fitter's single-topology path and the calculator agree per template."""
    _, reaction_set = water
    for template in reaction_set.data.molecules.values():
        atoms = template.atoms.copy()
        atoms.calc = None
        atoms.set_cell([14.0, 14.5, 15.0])
        atoms.center()
        atoms.pbc = periodic
        system = System(
            atoms,
            Topology.from_terms(template.terms, atoms),
            reaction_set,
            evb={"max_states": 1},
        )
        expected = system.calculate()
        got = evaluate(atoms, template.terms)
        assert got.lj == pytest.approx(expected["energy_lj"], abs=1e-10)
        assert got.energy == pytest.approx(expected["energy"], abs=1e-10)
        np.testing.assert_allclose(got.forces, expected["forces"], atol=1e-10)


def test_a_cutoff_charges_each_state_its_own_cut_sum(water):
    """Under `lj_cutoff`, each correction is that state's switched sum and tail.

    `_cut` sums the tail by atom type; the per-state path spreads it over pairs
    so that it can be averaged over states like the rest, and the two must
    still agree state by state.
    """
    root, reaction_set = water
    atoms, topology = zundel(root)
    atoms.pbc = True
    with use(dataclasses.replace(active(), lj_cutoff=5.5)):
        system = System(atoms, topology, reaction_set)
        results = system.calculate()
        (block,) = system.basis.build(atoms, system.topology, system.bimol_cutoff)
        lj = system.lj_ff
        assert lj.blocks[0].varies and lj.blocks[0].tail is not None

        brute = np.array([lj_of(atoms, state) for state in block.states])
        assert np.ptp(brute) > 1e-6
        np.testing.assert_allclose(lj.corrections(0), brute, rtol=0, atol=1e-12)
        weights = results["blocks"][0]["weights"]
        assert results["energy_lj"] == pytest.approx(weights @ brute, abs=1e-12)


# -- two blocks --------------------------------------------------------------


class TestTwoVaryingBlocks:
    """Two multi-state blocks that disagree, seeing each other through the 12-6.

    The cross term is the one place the weights of one block enter another's
    diagonal, which is what makes the 12-6 self-consistent; the finite
    differences below hold the Hellmann-Feynman forces and virial to it.
    """

    @pytest.fixture(params=[None, 7.0], ids=["all-pairs", "cutoff"])
    def case(self, water, request):
        root, reaction_set = water
        atoms, topology = two_transfers(root)
        # `lj_cutoff = 7` puts the pairs between the complexes in the switch.
        ff = dataclasses.replace(active(), lj_cutoff=request.param)

        def calculate(positions=None, cell=None):
            perturbed = atoms.copy()
            if positions is not None:
                perturbed.positions = positions
            if cell is not None:
                perturbed.set_cell(cell)
            with use(ff):
                system = System(perturbed, topology, reaction_set)
                return system.calculate(), system

        return atoms, calculate

    def test_the_cross_term_is_exercised(self, case):
        _, calculate = case
        results, system = calculate()
        assert [block["nstates"] for block in results["blocks"]] == [2, 2]
        assert system.lj_ff.self_consistent
        u, _ = system.lj_ff._cross_pairs(0, 1)
        cross = u.sum(axis=(2, 3))
        assert np.ptp(cross) > 1e-6, "the cross term does not depend on the states"

    def test_forces(self, case):
        atoms, calculate = case
        results, _ = calculate()
        h = 1e-5
        fd = np.zeros_like(atoms.positions)
        for i in range(len(atoms)):
            for k in range(3):
                pos = atoms.positions.copy()
                pos[i, k] += h
                plus = calculate(pos)[0]["energy"]
                pos[i, k] -= 2 * h
                minus = calculate(pos)[0]["energy"]
                fd[i, k] = -(plus - minus) / (2 * h)
        np.testing.assert_allclose(results["forces"], fd, atol=1e-6)

    def test_virial(self, case):
        atoms, calculate = case
        results, _ = calculate()
        w_fd = finite_difference_virial(
            lambda p, c: calculate(p, c)[0]["energy"],
            atoms.positions,
            np.asarray(atoms.cell),
            delta=1e-6,
        )
        np.testing.assert_allclose(results["virial"], w_fd, atol=1e-5)


# -- EVBSystem ---------------------------------------------------------------


def test_evb_system_charges_each_state_its_own_sum(water):
    """`EVBSystem` puts each state's difference from the first on the diagonal."""
    root, reaction_set = water
    atoms, topology = zundel(root)
    system = System(atoms, topology, reaction_set)
    system.calculate()
    # The block's states already carry their terms: the closure set them.
    (block,) = system.basis.build(atoms, system.topology, system.bimol_cutoff)
    states = block.states
    assert all(state.term_dict for state in states)

    def calculate(positions):
        perturbed = atoms.copy()
        perturbed.positions = positions
        return EVBSystem(perturbed, states, reaction_set=reaction_set).calculate()

    results = calculate(atoms.positions)
    brute = np.array([lj_of(atoms, state) for state in states])
    assert np.ptp(brute) > 1e-6
    assert results["energy_lj"] == pytest.approx(
        results["statevec"] @ brute, abs=1e-12
    )

    h = 1e-5
    fd = np.zeros_like(atoms.positions)
    for i in range(len(atoms)):
        for k in range(3):
            pos = atoms.positions.copy()
            pos[i, k] += h
            plus = calculate(pos)["energy"]
            pos[i, k] -= 2 * h
            minus = calculate(pos)["energy"]
            fd[i, k] = -(plus - minus) / (2 * h)
    np.testing.assert_allclose(results["forces"], fd, atol=1e-6)
