from ase import units
import numpy as np

from DynamicTopology.forcefield.ewald import Ewald, MinimumImage, contract_pairs
from DynamicTopology.forcefield.params import active


class ACKS2:
    """Charge-equilibration electrostatics.

    Two index spaces meet here and must not be confused.  The per-atom
    parameters (`mu`, `eta`, `soft_amp`, `soft_decay`) arrive in *term order* --
    the order the `atom` terms were collected -- while `pos` and the returned
    forces are in *global* atom order.  `indices[k]` is the global index of the
    k-th term.  Everything below is built in term order, and the forces are
    scattered back to global order only at the very end.

    Getting this wrong is silent: term order coincides with global order
    whenever the atom terms happen to be collected in index order, which is the
    common case, so the error only appears once a molecule is matched onto the
    live system in a different order.

    **The charge kernel is an object, not a matrix**, and everything that
    touches the geometry goes through it -- see `forcefield/ewald.py`.  Under
    open or partially periodic boundaries it is `MinimumImage`, the nearest-image
    `erf(2 r) / r` this class used to inline.  Under full periodicity it is
    `Ewald`, which sums that kernel over every image; the two differ in the
    energy, in the forces, and in the fact that the periodic `K_ii` is nonzero,
    because an atom does interact with its own images even though it does not
    interact with itself.  A method here that needs the kernel takes it as an
    optional argument and falls back to `MinimumImage`, so calling any of them
    with a bare `rij` still means the open-boundary problem.

    **The solve is topology-free and the energy is not.**  `build_system` never
    sees an exclusion: the charges come from the full kernel over every pair,
    which is what keeps them a function of the nuclear coordinates and the
    elements alone, hence the same on every diabatic state, hence solvable once
    per force call outside the EVB Hamiltonian.  The intramolecular exclusion
    enters afterwards, as the `screen` multiplier on the *energy* functional --
    `compute` and `compute_coulomb` -- and `System.calculate` assembles that
    screen from the ground-state weights of the states it has just
    diagonalized.  `forcefield/exclusions.py` has the argument for why it has to
    be this way round; `prepare`/`compute` is the split that makes it possible.
    """

    @property
    def CCOUL(self) -> float:
        """The Coulomb constant in eV*Angstrom, from the active parameters.

        A property rather than a class attribute so that it tracks the dataset
        rather than the import: fast-forces' `refine` builds one of these at
        module scope, before any manifest has been read.  `zbl` carries the same
        physical constant to more digits under its own field; see
        `params.ForceFieldParams.zbl_ccoul` for why the two are separate.
        """
        return active().ccoul

    def __init__(self):
        self.Q = None
        self.u = None
        self.A = None
        self.state_hash = None
        self.ewald = None
        self.ewald_key = None
        # Everything `compute` needs from the geometry `prepare` last saw.
        self.indices = None
        self.order = None
        self.kernel = None
        self.rij = None
        self.vecs = None
        self.params = None
        self.natoms = 0
        self.shape = None

    def build_system(self, rij, params, kernel=None):
        """Assemble the ACKS2 linear system `A x = b`, in term order.

        `x` is `[Q, u, lambda_total, lambda_KS]`: the charges, the Kohn-Sham
        potentials conjugate to them, and the two constraint multipliers.  `A`
        is symmetric, which is what lets the force adjoint below reuse it
        untransposed.

        Only two blocks of `A` depend on the geometry -- the Coulomb block and
        the softness block -- and `compute_response_forces` differentiates
        exactly those two.  Anything geometry-dependent added here must be
        differentiated there as well, or the forces stop being the gradient of
        the energy.

        The hardness is *added* to the Coulomb diagonal rather than overwriting
        it.  With open boundaries there is nothing there to overwrite, but the
        periodic kernel carries an atom's interaction with its own images on
        that diagonal, and it belongs in the equilibration alongside `2 eta`.

        **No exclusion reaches this matrix, deliberately.**  Masking the kernel
        here would be the more obviously consistent thing to do -- a screened
        kernel is simply another kernel, with energy, forces and virial all
        derived from it -- and it is exactly what makes the charges depend on
        the bond graph, which is what this term cannot afford.  See the class
        docstring and `forcefield/exclusions.py`.
        """
        natoms = rij.shape[0]
        neqns = 2 * natoms + 2
        diag = np.diag_indices(natoms)
        atom = np.arange(natoms)
        if kernel is None:
            kernel = MinimumImage(rij)

        A = np.zeros((neqns, neqns))
        b = np.zeros(neqns)

        # interaction
        A[:natoms, :natoms] = kernel.matrix()

        # softness
        amp, decay = params["soft_amp"], params["soft_decay"]
        X0 = amp[:, None] * amp[None, :]
        tau = 0.5 * (decay[:, None] + decay[None, :])
        bsoft = X0 * np.exp(-rij / tau)
        bsoft[diag] = 0.0
        bsoft[diag] = -1 * np.sum(bsoft, axis=1)
        A[natoms : 2 * natoms, natoms : 2 * natoms] = bsoft

        # coupling
        A[:natoms, natoms : 2 * natoms] = -np.eye(natoms)
        A[natoms : 2 * natoms, :natoms] = -np.eye(natoms)

        # diagonal
        A[atom, atom] += 2.0 * params["eta"]
        b[:natoms] = -params["mu"]

        # Constraints
        # KS coeffs
        A[-1, atom + natoms] = -1
        A[atom + natoms, -1] = -1
        b[-1] = 0.0

        # total charge
        A[-2, atom] = -1
        A[atom, -2] = -1
        b[-2] = 0.0

        return A, b

    def solve_charges(self, rij, params, kernel=None):
        """Charges, KS potentials and the system matrix.  All in term order."""
        natoms = rij.shape[0]
        A, b = self.build_system(rij, params, kernel)
        x = np.linalg.solve(A, b)
        return x[:natoms], x[natoms : 2 * natoms], A

    def compute_charges(self, rij, params, kernel=None):
        """Solve the ACKS2 linear system.  All arguments are in term order."""
        return self.solve_charges(rij, params, kernel)[0]

    def compute_coulomb(self, Q, rij, vecs, kernel=None, screen=None):
        """Coulomb energy, forces and virial.  All arguments and results in term order.

        The charges are held fixed here, so this is only the explicit part of
        the gradient.  `compute_response_forces` supplies the dQ/dr part, and
        `compute` adds the two; this method on its own is not the gradient of
        its own energy.

        `W` is the weight the kernel is contracted against: the energy is
        `sum_ij W_ij K_ij`, so `W` carries the 1/2 that halves the (i, j)/(j, i)
        double count as well as the unit conversion constant, and the kernel
        returns `dS/dr` and `dS/de` for that same sum.  The diagonal is not
        masked off -- `K_ii` is zero under open boundaries and is a real
        self-image interaction under periodic ones.

        `screen` is the intramolecular exclusion: an `(n, n)` symmetric
        multiplier, 1 on a pair the kernel should act between and 0 on one a
        template has claimed for its bonded terms.  It multiplies the weight,
        which is the same thing as multiplying the kernel: the energy is
        `sum_ij W_ij K_ij` and `contract` differentiates that same sum, so a
        zero here removes the pair from the energy, the force and the virial
        together with no separate correction to keep in step.

        **Entries between 0 and 1 are meaningful and are the normal case.**  The
        screen `System.calculate` builds is `1 - sum_s w_s M_s` over the states
        of an EVB block, `w_s` their ground-state weights: a pair bonded in
        every state of a block is screened out entirely, and one bonded in only
        some of them is screened by however much ground state those states hold.
        That is not an interpolation invented here -- it is what the
        Hellmann-Feynman contraction of a per-state diagonal correction comes
        to, term by term, and it is why one contraction serves a whole block.
        """
        if kernel is None:
            kernel = MinimumImage(rij, vecs)

        W = 0.5 * self.CCOUL * (Q[:, None] * Q[None, :])
        if screen is not None:
            W = W * screen
        e_tot = np.sum(W * kernel.matrix()) * units.eV

        dS_dr, dS_de = kernel.contract(W)
        f_tot = -dS_dr * units.eV / units.Angstrom

        # Virial at fixed `Q`, the explicit half of the strain derivative, in
        # the same relationship to `e_tot` as `f_tot` is.
        w_tot = dS_de * units.eV
        return e_tot, f_tot, w_tot

    def compute_response_forces(
        self, Q, u, A, rij, vecs, params, kernel=None, screen=None
    ):
        """The dQ/dr part of the force, and its virial.  All arguments and results in term order.

        The charges are not independent of the geometry: they solve `A(r) x = b`
        with `b` geometry-free, so moving an atom moves every charge.  The
        energy `E = CCOUL/2 * Q.K.Q` is not stationary in `Q` -- the ACKS2
        functional is, but this Coulomb piece alone is not -- so that motion
        contributes to `dE/dr` and cannot be dropped.  Omitting it is what made
        the electrostatic force disagree with its own energy for every species
        with nonzero charges, and NVE energy drift the symptom.

        Differentiating the solve gives `dx/dr = -A^-1 (dA/dr) x`, so

            dE/dr = (dE/dr)|_Q  +  (dE/dx) . dx/dr
                  = (dE/dr)|_Q  -  lam^T (dA/dr) x,   lam = A^-1 (dE/dx)

        which needs one extra solve rather than one per coordinate.  `A` is
        symmetric, so no transpose is required.

        `dA/dr` is nonzero only in the two blocks `build_system` builds from the
        geometry, and each is differentiated by contracting it against the
        symmetric weight matrix that `-lam^T (dA/dr) x` puts on it.  The
        softness block needs care: `X_ii = -sum_j X_ij`, so each off-diagonal
        `bsoft_ij` appears in four entries of `X` and all four contribute.

        **`screen` belongs to `dE/dx` and to nothing else here, and the
        asymmetry is the whole point.**  The energy is the screened one, so its
        derivative with respect to the charges carries the screen.  `A` is the
        *unscreened* matrix -- `build_system` never saw an exclusion -- so
        `dA/dr` is the unscreened kernel derivative and the weight it is
        contracted against must not be masked.  Screening both would be the
        gradient of a functional whose charges were also screened, which is not
        the functional being evaluated; screening neither drops the exclusion
        from the response entirely.  Either mistake shows up only as NVE drift.
        """
        natoms = len(Q)
        diag = np.diag_indices(natoms)
        r = rij + np.finfo(np.float64).eps
        if kernel is None:
            kernel = MinimumImage(rij, vecs)

        # dE/dx, nonzero only on the charge block.  The multiplier rows are
        # geometry-free and the energy does not depend on u.
        gradient = np.zeros(A.shape[0])
        K = kernel.matrix()
        if screen is not None:
            K = K * screen
        gradient[:natoms] = self.CCOUL * (K @ Q)
        lam = np.linalg.solve(A, gradient)
        lam_q, lam_u = lam[:natoms], lam[natoms : 2 * natoms]

        # -lam^T (dA/dr) x for the Coulomb block, as a weight on the kernel.
        # Entry (i, j) and entry (j, i) each hold the whole pair term, so the
        # weight is halved to match the convention `compute_coulomb` uses.
        # Unscreened, for the reason in the docstring.
        W = -0.5 * (lam_q[:, None] * Q[None, :] + lam_q[None, :] * Q[:, None])
        coulomb_dr, coulomb_de = kernel.contract(W)

        # The same for the softness block, which stays a nearest-image pair
        # term at every boundary condition -- see `ewald.py`.
        amp, decay = params["soft_amp"], params["soft_decay"]
        tau = 0.5 * (decay[:, None] + decay[None, :])
        dbsoft_dr = -(amp[:, None] * amp[None, :]) * np.exp(-rij / tau) / tau
        dbsoft_dr[diag] = 0.0
        W_soft = -0.5 * (
            lam_u[:, None] * u[None, :]
            + lam_u[None, :] * u[:, None]
            - lam_u[:, None] * u[:, None]
            - lam_u[None, :] * u[None, :]
        )
        soft_dr, soft_de = contract_pairs(W_soft * dbsoft_dr, vecs, r)

        # Both blocks are contracted as `dS/dr`; the force is minus that.
        return -(coulomb_dr + soft_dr), coulomb_de + soft_de

    def get_kernel(self, pos, vecs, rij, pbc, cell):
        """The charge kernel for these boundary conditions, in term order.

        Ewald needs all three directions periodic; a slab or a wire keeps the
        nearest-image kernel, which is what it had before periodic
        electrostatics existed here.  The cell-dependent half of the Ewald setup
        -- the splitting parameter and the reciprocal vectors -- is cached, so a
        fixed cell builds it once and an NPT trajectory rebuilds it per step.
        """
        if not np.all(pbc):
            return MinimumImage(rij, vecs)

        cell = np.asarray(cell, dtype=float)
        key = cell.tobytes()
        if self.ewald is None or key != self.ewald_key:
            self.ewald = Ewald(cell)
            self.ewald_key = key
        return self.ewald.bind(pos, vecs, rij)

    # -- the intramolecular exclusion ---------------------------------------

    def _local(self, pairs):
        """`pairs`, global indices, mapped into term order.

        Conflating the two orders is the silent failure the class docstring
        warns about: they coincide whenever the atom terms were collected in
        index order, which is the common case, so a mistake here shows up only
        once a template is matched onto a live system in a different order.
        """
        pairs = np.asarray(pairs, dtype=int)
        if pairs.size == 0:
            return pairs.reshape(0, 2)
        local = self.order[pairs[:, :2]]
        if np.any(local < 0):
            raise KeyError(
                "A `coulombexclusion` term names an atom with no `atom` term; "
                "the exclusion and the charge equilibration disagree about "
                "which atoms exist."
            )
        return local

    def exclusion_energy(self, pairs) -> float:
        """What removing `pairs` from the Coulomb sum is worth, in eV.

        `E = CCOUL sum_{i<j} Q_i Q_j K_ij` over the whole system, so dropping a
        pair costs exactly `-CCOUL Q_i Q_j K_ij` -- one lookup per pair into the
        kernel matrix `prepare` has already built and memoized.  That is the
        whole cost of putting this correction on a diabatic state's diagonal,
        which is what makes it affordable to evaluate per state where the
        charges themselves are not.

        Requires `prepare` to have run at the current geometry; the charges it
        reads are the unscreened ones, identically so for every state.
        """
        local = self._local(pairs)
        if len(local) == 0:
            return 0.0
        i, j = local[:, 0], local[:, 1]
        K = self.kernel.matrix()
        return -self.CCOUL * float(np.sum(self.Q[i] * self.Q[j] * K[i, j])) * units.eV

    def screen_matrix(self, weighted_pairs):
        """`1 - sum_s w_s M_s`, in term order, or None if nothing is excluded.

        `weighted_pairs` is a list of `(pairs, weight)`: one entry per diabatic
        state, `pairs` its excluded pairs in global indices and `weight` its
        ground-state weight.  Blocks are atom-disjoint and a block's weights sum
        to one, so a pair bonded in every state of its block comes out at
        exactly 0 and the arithmetic never runs past it.

        This is the object `compute_coulomb`'s docstring calls a fractional
        screen, and it is not an approximation: it is the Hellmann-Feynman
        contraction `sum_s w_s dE_s/dr` written as a single weight matrix, which
        is legitimate because every state's correction is linear in its own mask
        and the kernel contraction is linear in the weight.  One contraction and
        one adjoint solve therefore serve the whole system, where a literal
        per-state evaluation would cost one of each per diabatic state.
        """
        if not weighted_pairs:
            return None
        screen = np.ones((self.natoms, self.natoms))
        for pairs, weight in weighted_pairs:
            local = self._local(pairs)
            if len(local) == 0:
                continue
            i, j = local[:, 0], local[:, 1]
            np.subtract.at(screen, (i, j), weight)
            np.subtract.at(screen, (j, i), weight)
        return screen

    def screen_from_terms(self, term_dict):
        """The screen a single topology's `coulombexclusion` terms describe.

        The one-state case of `screen_matrix`, for every caller that evaluates
        one fixed bonding pattern rather than an EVB ground state: `evb.py`,
        fast-forces' `refine`, and `__call__` itself.  Returns None when a term
        list carries no exclusions, so a dataset that predates them evaluates
        exactly the unscreened kernel it always did.
        """
        params = term_dict.get("coulombexclusion")
        if params is None:
            return None
        return self.screen_matrix([(params["atoms"], 1.0)])

    # -- evaluation ----------------------------------------------------------

    def prepare(self, pos, pbc, cell, term_dict: dict) -> np.ndarray:
        """Solve the charges at this geometry and cache what `compute` needs.

        Separated from `compute` because the charges are available before the
        exclusion is: `System.calculate` cannot know which pairs to screen until
        it has diagonalized the EVB blocks, and it cannot build those blocks'
        diagonal corrections without the charges.  Nothing here depends on the
        bond graph, which is exactly why the order works.

        Returns the term-order-to-global index array, for a caller that needs to
        talk about atoms in term order.
        """
        pbc = np.asarray(pbc, dtype=bool)
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell
        atom_params = term_dict.get("atom")
        if atom_params is None:
            raise KeyError("No atom parameters set")
        indices = atom_params["atoms"][:, 0]
        params = atom_params["kwargs"]

        # Both axes in term order, so the parameter vectors line up with them.
        sub = np.ix_(indices, indices)
        vecs = vecs[sub]
        rij = np.sqrt(np.sum(vecs * vecs, -1))
        kernel = self.get_kernel(pos[indices], vecs, rij, pbc, cell)

        order = np.full(int(indices.max()) + 1, -1, dtype=int)
        order[indices] = np.arange(len(indices))

        self.indices = indices
        self.order = order
        self.kernel = kernel
        self.rij = rij
        self.vecs = vecs
        self.params = params
        self.natoms = len(indices)
        self.shape = pos.shape

        # Cache on the parameters and the cell as well as the geometry: the same
        # positions with a different set of atom terms is a different problem,
        # and so is the same system in a cell a barostat has just rescaled.  The
        # bonding is deliberately absent from the key -- the solve does not
        # depend on it, and a key that pretended otherwise would throw the
        # charges away after every reaction for nothing.
        state_hash = hash(
            (
                pos.tobytes(),
                indices.tobytes(),
                np.asarray(cell, dtype=float).tobytes(),
            )
        )
        if self.Q is None or state_hash != self.state_hash:
            self.Q, self.u, self.A = self.solve_charges(rij, params, kernel)
            self.state_hash = state_hash
        return indices

    def compute(self, screen=None) -> tuple[float, np.ndarray, np.ndarray]:
        """Energy, forces and virial of the screened Coulomb functional.

        All three are derived from the one functional
        `E = CCOUL/2 sum_ij S_ij Q_i Q_j K_ij`, so they are consistent with each
        other whatever `S` is -- including the fractional screen an EVB ground
        state produces.  `S` is held fixed under the derivative, which is
        correct and not an omission: the weights it is built from are
        eigenvector components, and a Hellmann-Feynman gradient does not
        differentiate those.
        """
        e_tot, f_tot, w_tot = self.compute_coulomb(
            self.Q, self.rij, self.vecs, self.kernel, screen
        )
        f_resp, w_resp = self.compute_response_forces(
            self.Q,
            self.u,
            self.A,
            self.rij,
            self.vecs,
            self.params,
            self.kernel,
            screen,
        )
        f_tot = f_tot + f_resp
        w_tot = w_tot + w_resp

        # scatter term-ordered forces back to global atom order.  The virial
        # needs no scatter: it is a single 3x3 sum over pairs, not a per-atom
        # quantity, so term order and global order give the same matrix.
        forces = np.zeros(self.shape)
        forces[self.indices] = f_tot
        return e_tot, forces, w_tot

    # -- the block interface `System.calculate` drives -----------------------
    #
    # Shared with `pointcharge.PointCharge`, so `System` diagonalizes against
    # either without branching.  Here it is bookkeeping over the methods above:
    # the corrections are the per-state exclusion energies, fixed once the
    # charges are, and `evaluate` is the screened `compute`.

    # The corrections do not depend on any other block's weights, so one pass
    # over the blocks is already self-consistent.
    self_consistent: bool = False

    def bind(self, blocks) -> None:
        """Each state's `coulombexclusion` pairs and what they are worth.

        Global atom indices, straight off the state's own term dict --
        `EVBBasis` has already given every state in a block its terms in order
        to evaluate its bonded energy.  A dataset that ships no
        `coulombexclusion` terms yields nothing and the electrostatics are
        unscreened, exactly as before they existed.
        """
        self.block_pairs = []
        self.block_corrections = []
        self.block_weights = []
        for block in blocks:
            pairs = []
            for state in block.states:
                params = state.term_dict.get("coulombexclusion")
                pairs.append(
                    np.zeros((0, 2), dtype=int)
                    if params is None
                    else params["atoms"][:, :2]
                )
            self.block_pairs.append(pairs)
            self.block_corrections.append(
                np.array([self.exclusion_energy(p) for p in pairs])
            )
            weights = np.zeros(block.nstates)
            weights[block.seed_index] = 1.0
            self.block_weights.append(weights)

    def corrections(self, index: int) -> np.ndarray:
        return self.block_corrections[index]

    def update(self, index: int, weights: np.ndarray) -> None:
        self.block_weights[index] = weights

    def evaluate(self) -> tuple[float, np.ndarray, np.ndarray]:
        """`compute` under the screen the current weights describe.

        The screen is `1 - sum_s w_s M_s`: a pair intramolecular in every state
        of its block drops out entirely, one intramolecular in only some of them
        drops out by however much ground state those states hold.  That is
        `sum_s w_s d(correction_s)/dr` written as one weight matrix, which the
        linearity of the kernel contraction permits.
        """
        weighted_pairs = [
            (pairs, float(w))
            for block_pairs, weights in zip(self.block_pairs, self.block_weights)
            for pairs, w in zip(block_pairs, weights)
        ]
        return self.compute(self.screen_matrix(weighted_pairs))

    def __call__(
        self, pos, pbc, cell, term_dict: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """One fixed bonding pattern, screened by its own exclusion terms.

        The whole-system EVB path does not come through here -- `System` calls
        `prepare` and `compute` around its diagonalization, because the screen
        it wants is not any single topology's.  This is the honest statement of
        what the term is for one topology, and what every caller outside the EVB
        wants.
        """
        self.prepare(pos, pbc, cell, term_dict)
        return self.compute(self.screen_from_terms(term_dict))
