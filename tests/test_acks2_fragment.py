"""Fragment ACKS2 and the electrostatic admission gate, against values.

`test_gradients.py` and `test_stress.py` hold the forces and the virial to
finite differences; a derivative can be right of the wrong energy.  These pin
the energy itself: that the per-state Schur solve is the full solve, that a
molecule keeps its formal charge, that a lone template scores zero, that the
zero-softness limit is the point-charge term, and -- the reason the scheme
exists -- that the surrounding charges decide which way a proton transfer goes.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.params import ForceFieldParams, use
from DynamicTopology.forcefield.pointcharge import PointCharge
from DynamicTopology.system import System

from test_stress import finite_difference_virial
from test_gradients import (
    CELL,
    PBC,
    POS_ZUNDEL,
    ZUNDEL_A,
    ZUNDEL_B,
    ZUNDEL_SYMBOLS,
    acks2_term_dict,
    fake_block,
    finite_difference_forces,
    zundel_state,
    zundel_system,
)

BOUNDARIES = [(PBC, CELL), (np.ones(3, dtype=bool), np.eye(3) * 9.0)]
BOUNDARY_IDS = ["open", "pbc"]


@pytest.mark.parametrize(("pbc", "cell"), BOUNDARIES, ids=BOUNDARY_IDS)
def test_the_schur_solve_is_the_full_solve(pbc, cell):
    """Each state's diagonal entry is its whole-system minimum, to rounding.

    The block solve folds the environment in as a potential and a reaction
    field; solving the same state over every atom at once must give the same
    energy.  The environment's own terms -- its constant and its isolated
    references -- are shared by every state and are what `evaluate` adds once.
    """
    acks2 = ACKS2()
    acks2.prepare(POS_ZUNDEL, pbc, cell, zundel_system(ZUNDEL_A))
    acks2.bind([fake_block([zundel_state(ZUNDEL_A), zundel_state(ZUNDEL_B)])])
    corrections = acks2.corrections(0)
    shared = acks2.CCOUL * (acks2.const_env - acks2.iso_env[0])

    for s, state in enumerate((ZUNDEL_A, ZUNDEL_B)):
        full = ACKS2()(POS_ZUNDEL, pbc, cell, zundel_system(state))[0]
        assert corrections[s] + shared == pytest.approx(full, abs=1e-10)


@pytest.mark.parametrize(("pbc", "cell"), BOUNDARIES, ids=BOUNDARY_IDS)
def test_every_molecule_keeps_its_formal_charge(pbc, cell):
    """Softness acts within a molecule, so no charge crosses between two.

    The hydronium holds +1 and each water 0 to machine precision, however
    strongly the ion polarizes its neighbours.
    """
    acks2 = ACKS2()
    acks2(POS_ZUNDEL, pbc, cell, zundel_system(ZUNDEL_A))
    q = acks2.Q
    np.testing.assert_allclose(
        [q[[0, 1, 2]].sum(), q[[3, 4, 5, 6]].sum(), q[[7, 8, 9]].sum()],
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )
    # Anti-vacuity: the waters are polarized, not merely neutral.
    assert np.abs(q[[0, 1, 2]]).max() > 0.1


def test_a_lone_template_scores_zero():
    """The isolated-molecule reference is the molecule itself, so nothing is left.

    Its gas-phase electrostatics belong to its bonded terms -- the property the
    Coulomb exclusion used to provide -- and here it holds exactly, charges and
    all, where the exclusion held it by removing the charges' own interaction.
    """
    td = acks2_term_dict(ZUNDEL_SYMBOLS[:3], [(0, 1), (0, 2)])
    acks2 = ACKS2()
    energy, forces, virial = acks2(POS_ZUNDEL[:3], PBC, CELL, td)
    assert energy == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(forces, 0.0, atol=1e-12)
    np.testing.assert_allclose(virial, 0.0, atol=1e-12)
    assert np.abs(acks2.Q).max() > 0.1, "the water carries no charge to cancel"
    assert abs(acks2.CCOUL * acks2.iso_env[0]) > 1.0, (
        "the isolated reference is negligible, so a lone template scoring zero "
        "proves nothing about the subtraction"
    )


@pytest.mark.parametrize(("pbc", "cell"), BOUNDARIES, ids=BOUNDARY_IDS)
def test_zero_softness_is_the_point_charge_term(pbc, cell):
    """With `X = 0` the charges are `q0`, and the energy is `PointCharge`'s.

    `mu` and `eta` are zeroed too, since they would otherwise add a
    geometry-free self-energy per state.  `PointCharge` excludes every
    intramolecular pair here -- all within `exclusion_depth` -- which is what
    subtracting the isolated references does.  Under Ewald each molecule still
    sees its own images in both.
    """
    molecules = {
        "a": ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9]),
        "b": ([0, 1, 2, 6], [3, 4, 5], [7, 8, 9]),
    }
    for state, key in ((ZUNDEL_A, "a"), (ZUNDEL_B, "b")):
        td = zundel_system(state)
        kwargs = td["atom"]["kwargs"]
        for name in ("mu", "eta", "soft_amp"):
            kwargs[name] = np.zeros_like(kwargs[name])
        td["charge"] = {"atoms": td["atom"]["atoms"], "kwargs": {"q": kwargs["q0"]}}
        pairs = [(i, j) for mol in molecules[key] for i in mol for j in mol if i < j]
        td["coulombexclusion"] = {"atoms": np.array(pairs), "kwargs": {}}

        e_a, f_a, w_a = ACKS2()(POS_ZUNDEL, pbc, cell, td)
        e_p, f_p, w_p = PointCharge()(POS_ZUNDEL, pbc, cell, td)
        assert e_a == pytest.approx(e_p, abs=1e-10)
        np.testing.assert_allclose(f_a, f_p, atol=1e-10)
        np.testing.assert_allclose(w_a, w_p, atol=1e-10)


class TestTheEnvironmentDecidesTheTransfer:
    """The point of per-state charges: solvation biases which diabat is lower.

    A hydroxide sits beside the Zundel, near one oxygen or the other.  The +1
    is stabilized on whichever side the anion is, so the diagonal gap between
    the two proton positions has to change sign when the anion moves across.
    The ACKS2 this replaced gave both states the same charges, and the gap was
    the bonded one alone.
    """

    @staticmethod
    def gap(anion_near):
        pos = np.vstack(
            [
                POS_ZUNDEL[:7],
                [anion_near + [0.0, 2.8, 0.0], anion_near + [0.3, 3.7, 0.0]],
            ]
        )
        symbols = ZUNDEL_SYMBOLS[:7] + ["O", "H"]
        bonds_a, q0_a = ZUNDEL_A
        seed = acks2_term_dict(symbols, bonds_a + [(7, 8)], q0_a + [-1.0, 0.0])
        acks2 = ACKS2()
        acks2.prepare(pos, PBC, CELL, seed)
        acks2.bind([fake_block([zundel_state(ZUNDEL_A), zundel_state(ZUNDEL_B)])])
        corrections = acks2.corrections(0)
        # State B puts the +1 on O0, state A on O3.
        return corrections[1] - corrections[0]

    def test_the_gap_follows_the_anion(self):
        near_o0 = self.gap(POS_ZUNDEL[0])
        near_o3 = self.gap(POS_ZUNDEL[3])
        assert near_o0 < -0.1, f"anion by O0 favours the proton there by {near_o0}"
        assert near_o3 > 0.1, f"anion by O3 favours the proton there by {-near_o3}"


# -- the admission gate, under the dataset whose charges it screens with -------

MANIFEST = Path("datasets/Water-fixed-pc/Water.json").resolve()
ZUNDEL_FRAME = Path("datasets/Water-fixed-pc/reactions/h3o-h2o-transfer.xyz")
WATER = Path("datasets/Water-fixed-pc/molecules/h2o.xyz")
PARAMS = ForceFieldParams.from_dict(json.loads(MANIFEST.read_text())["global_params"])


@pytest.fixture(scope="module")
def pointcharge_set():
    with use(PARAMS):
        yield ReactionSet(MANIFEST)


class TestTheElectrostaticGap:
    """`ElectrostaticGap` is the diagonal's own electrostatic difference.

    Under `pointcharge` the charges it screens with are the ones the diagonal
    uses, and with one multi-state block the environment it holds fixed is the
    one the diagonal sees -- so the two numbers must agree exactly, and must be
    antisymmetric in the two diabats for the closure's seed-invariance.
    """

    @staticmethod
    def system(reaction_set, pbc=False):
        zundel = io.read(ZUNDEL_FRAME, index=1)
        zundel.calc = None
        zundel.positions[1, 0] += 0.05
        spectator = io.read(WATER)
        spectator.calc = None
        spectator.positions += [0.4, 3.1, 0.3]
        connectivity = list(zundel.info["connectivity"]) + [
            [i + len(zundel), j + len(zundel), order]
            for i, j, order in spectator.info["connectivity"]
        ]
        atoms = zundel + spectator
        atoms.info["connectivity"] = connectivity
        atoms.set_cell(np.eye(3) * (12.0 if pbc else 20.0))
        atoms.pbc = pbc
        system = System(atoms, Topology.from_atoms(atoms), reaction_set)
        results = system.calculate()
        return system, results

    def test_the_gap_is_the_diagonal_difference(self, pointcharge_set):
        with use(PARAMS):
            system, _ = self.system(pointcharge_set)
            blocks = system.basis.build(system.atoms, system.topology, 4.0)
            multi = [k for k, block in enumerate(blocks) if block.nstates > 1]
            assert len(multi) == 1
            block = blocks[multi[0]]
            ff = system.nonbonded_ff
            ff.bind(blocks)
            corrections = ff.corrections(multi[0])

            a, b = block.states[0], block.states[1]
            forward = system.gap.energy(a, a.graph, b.graph)
            backward = system.gap.energy(b, b.graph, a.graph)
            assert forward == pytest.approx(-backward, abs=1e-12)
            assert forward == pytest.approx(corrections[1] - corrections[0], abs=1e-10)
            assert abs(forward) > 1e-3, "the spectator does not bias the hop at all"

    @pytest.mark.parametrize("pbc", [False, True], ids=["open", "pbc"])
    def test_the_gap_gradient(self, pointcharge_set, pbc):
        """What the switch's force term differentiates, against the gap itself.

        Needed only for a channel inside the admission ramp, and HCombustion's
        reference charges are all zero, so nothing else in the suite reaches it.
        """
        with use(PARAMS):
            system, _ = self.system(pointcharge_set, pbc)
            blocks = system.basis.build(system.atoms, system.topology, 4.0)
            block = next(block for block in blocks if block.nstates > 1)
            a, b = block.states[0], block.states[1]
            ff, gap = system.nonbonded_ff, system.gap
            td = system.topology.term_dict
            atoms = system.atoms

            def energy(p, cell=atoms.cell.array):
                ff.prepare(p, atoms.pbc, cell, td)
                gap.bind(ff, td)
                return gap.energy(a, a.graph, b.graph)

            energy(atoms.positions)
            forces, virial = gap.gradients(a, a.graph, b.graph)
            fd = finite_difference_forces(energy, atoms.positions.copy())
            np.testing.assert_allclose(forces, fd, atol=1e-6, rtol=1e-5)
            if pbc:
                w_fd = finite_difference_virial(
                    energy, atoms.positions.copy(), atoms.cell.array
                )
                np.testing.assert_allclose(virial, w_fd, atol=1e-6, rtol=1e-4)


# -- through `System`, on real templates carrying formal reference charges -----

WATER_MANIFEST = Path("datasets/Water/Water.json").resolve()
WATER_PARAMS = ForceFieldParams.from_dict(
    json.loads(WATER_MANIFEST.read_text())["global_params"]
)
# Formal charges per template, the ions' spread over their atoms; the shipped
# `atom` terms state none yet, which scores the ions neutral.
FORMAL_Q0 = {"H3O": [-0.5, 0.5, 0.5, 0.5], "HO": [-1.2, 0.2]}


@pytest.fixture(scope="module")
def charged_water():
    """The Water set with `q0` put into every `atom` term, in memory."""
    with use(WATER_PARAMS):
        reaction_set = ReactionSet(WATER_MANIFEST)
        for template in reaction_set.data.molecules.values():
            symbols = template.atoms.get_chemical_symbols()
            q0 = FORMAL_Q0.get(template.atoms.get_chemical_formula(mode="hill"))
            for term in template.terms:
                if term["type"] != "atom":
                    continue
                (atom,) = term["atoms"].values()
                if q0 is None:
                    term["kwargs"]["q0"] = 0.0
                else:
                    # O first, then the hydrogens, whatever the template order.
                    heavy = symbols[atom] == "O"
                    term["kwargs"]["q0"] = q0[0] if heavy else q0[1]
        reaction_set._term_cache.clear()
        yield reaction_set


def charged_zundel(proton=0.05, pbc=False):
    frame = io.read(Path("datasets/Water/reactions/h3o-h2o-transfer.xyz"), index=1)
    frame.calc = None
    frame.positions[1, 0] += proton
    spectator = io.read(Path("datasets/Water/molecules/h2o.xyz"))
    spectator.calc = None
    spectator.positions += [0.4, 3.1, 0.3]
    connectivity = list(frame.info["connectivity"]) + [
        [i + len(frame), j + len(frame), order]
        for i, j, order in spectator.info["connectivity"]
    ]
    atoms = frame + spectator
    atoms.info["connectivity"] = connectivity
    atoms.set_cell(np.eye(3) * 12.0)
    atoms.pbc = pbc
    atoms.positions += np.random.default_rng(3).normal(0.0, 0.02, atoms.positions.shape)
    return atoms


class TestThroughSystem:
    """Fragment ACKS2 with charged diabats, real blocks and the gate switched on."""

    def test_the_hop_carries_the_charge(self, charged_water):
        with use(WATER_PARAMS):
            atoms = charged_zundel()
            system = System(atoms, Topology.from_atoms(atoms), charged_water)
            results = system.calculate()
            multi = [b for b in results["blocks"] if b["nstates"] > 1]
            assert len(multi) == 1 and results["electrostatics_sweeps"] == 1
            assert system.gap.enabled
            q = system.nonbonded_ff.Q
            # The +1 is shared between the two oxygens' molecules in proportion
            # to the weights, and the spectator stays neutral.
            assert q[-3:].sum() == pytest.approx(0.0, abs=1e-10)
            assert q.sum() == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("pbc", [False, True], ids=["open", "pbc"])
    def test_forces(self, charged_water, pbc):
        from test_gradients import finite_difference_forces as fd_forces

        with use(WATER_PARAMS):
            atoms = charged_zundel(pbc=pbc)
            topology = Topology.from_atoms(atoms)

            def energy(p):
                probe = atoms.copy()
                probe.positions = p
                return System(probe, topology, charged_water).calculate()["energy"]

            forces = System(atoms, topology, charged_water).calculate()["forces"]
            np.testing.assert_allclose(
                forces, fd_forces(energy, atoms.positions.copy()), atol=1e-5, rtol=1e-5
            )
