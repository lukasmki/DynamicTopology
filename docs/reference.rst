Equation reference
===================

:doc:`concepts` describes how a force call puts these terms together.

The model is a multi-state valence-bond Hamiltonian. Each state :math:`s` is
one bonding pattern (one bond graph) over the same set of nuclei. The diagonal
:math:`H_{ss}` is the potential energy of that bonding pattern, and the
off-diagonal :math:`H_{st}` couples two of them. The physical energy is the
lowest eigenvalue of :math:`H`, and forces follow by Hellmann–Feynman.

.. math::

   H_{ss} &= E_\text{bonded}(s) - E_\text{excl}(s) + E_\text{elec}(s)
            + E_\text{ZBL} + E_\text{12-6} \\
   H_{st} &= V(\mathbf{x}) \qquad \text{one Gaussian per reaction channel}

:math:`E_\text{bonded}`, :math:`E_\text{excl}` and :math:`E_\text{elec}` depend
on the bonding pattern. The two repulsion sums :math:`E_\text{ZBL}` and
:math:`E_\text{12-6}` are functions of the nuclear positions and atomic numbers
alone, so they take the same value on every state and are added outside the
Hamiltonian.


Units and conventions
---------------------

.. list-table::
   :header-rows: 1

   * - quantity
     - unit
   * - length
     - Å
   * - energy
     - eV
   * - angle
     - radians
   * - charge
     - e
   * - force
     - eV/Å
   * - stress
     - eV/Å³

Dimensionless: ``soft_core``, ``n``, ``accuracy``. ``SCREENING_LENGTH`` is Å,
``ccoul`` is eV·Å and ``gamma`` is 1/Å. A coupling amplitude :math:`A` is eV and
its width :math:`a` is 1/Å²; :math:`r_0`, :math:`r_{a0}`, :math:`r_{b0}` are Å
and :math:`t_0` is radians.

Every parameter is held, evaluated and stored (in a ``.jsonl``) in these units.
q-force and OpenMM state the bonded and 12-6 parameters in nm and kJ/mol, and
``io/units.py`` is the table the conversion at that boundary goes through.

Gradient conventions, with :math:`\mathbf{v}` any interatomic displacement
vector:

.. math::

   F_i = -\frac{\partial E}{\partial \mathbf{r}_i}, \qquad
   W_{ab} = \sum_{\mathbf{v}} v_a \frac{\partial E}{\partial v_b}, \qquad
   \sigma = W / V

Every diagonal energy is a function of minimum-image displacement vectors
alone, so a homogeneous strain acts as :math:`\mathbf{v} \to (I + e)\mathbf{v}`
and the virial follows from the same pair gradients as the forces. A coupling
is written over absolute positions, and its virial is
:math:`W_{ab} = -\sum_i r_{i,a} F_{i,b}`. This does not depend on the origin,
because every coupling's forces sum to zero.


1. Bonded terms
---------------

Per-molecule terms defined on the bond graph of a state. Throughout,
:math:`r` is a bond length, :math:`\theta` a bond angle, :math:`\varphi` a
dihedral, and

.. math::

   \delta r = r - r_0, \qquad \delta c = \cos\theta - \cos\theta_0

Angles enter only through their cosines, never as angles.

1.1 Bond: Morse with a one-sided asymptote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   D_w &= \begin{cases} D + h & \delta r > 0 \\ D & \delta r \le 0 \end{cases} \\
   a   &= \sqrt{k / (2 D_w)} \\
   E   &= D_w \left[ 1 - e^{-a\,\delta r} \right]^2 - D

Parameters: :math:`D` (well depth), :math:`r_0` (equilibrium length),
:math:`k` (force constant), :math:`h` (asymptote height; optional, defaulting to
the global ``bond_asymptote``).

- The :math:`-D` offset puts the minimum at :math:`-D`, so the dissociated
  limit sits at :math:`h` above zero.
- The join at :math:`\delta r = 0` is C²: the curvature there is
  :math:`2 D_w a^2 = k` for either value of :math:`D_w`, so a fitted frequency
  sees neither the join nor :math:`h`.
- At fixed :math:`k`, raising :math:`h` lifts the whole stretched branch
  monotonically towards the harmonic :math:`k\,\delta r^2/2`, and the curve stays
  a Morse.

1.2 Bond: harmonic (non-reactive alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   E = \tfrac12 k\,\delta r^2 \qquad (D \text{ unused})

1.3 Angle: cosine harmonic
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   E = \tfrac12 k (\cos\theta - \cos\theta_0)^2

1.4 Cross terms
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - term
     - energy
     - parameters
     - atoms
   * - bond–bond
     - :math:`\max(k\,\delta r_1\,\delta r_2,\ -0.10364\ \text{eV})`
     - ``r1_0, r2_0, k``
     - 4 (two bonds)
   * - bond–angle
     - :math:`\max(k\,\delta r\,\delta c,\ -0.20728\ \text{eV})`
     - ``theta0, r0, k``
     - 5 (angle + bond)
   * - angle–angle
     - :math:`k\,\delta c_1\,\delta c_2`
     - ``theta1_0, theta2_0, k``
     - 6 (two angles)

The two floors are q-force's −10 and −20 kJ/mol, applied to the raw product.
The gradient is zero wherever a floor is active.

1.5 Dihedrals
~~~~~~~~~~~~~

The dihedral about a central bond :math:`\mathbf{v}_b` is
:math:`\varphi = \operatorname{atan2}(S, C)` with

.. math::

   \hat{\mathbf{n}} = \mathbf{v}_b / |\mathbf{v}_b|, \qquad
   S = (\hat{\mathbf{n}} \times \mathbf{u}) \cdot \mathbf{w}, \qquad
   C = \mathbf{u} \cdot \mathbf{w}

where :math:`\mathbf{u}` and :math:`\mathbf{w}` are the outer bond vectors
projected perpendicular to :math:`\hat{\mathbf{n}}`. Writing
:math:`P = 1 + \cos(n\varphi - \varphi_0)`:

.. list-table::
   :header-rows: 1

   * - term
     - energy
     - parameters
     - atoms
   * - periodic dihedral
     - :math:`k P`
     - ``phi0, n, k``
     - 4
   * - dihedral–bond
     - :math:`k\,\delta r\,P`
     - ``+ r0``
     - 6
   * - dihedral–angle
     - :math:`k\,\delta c\,P`
     - ``+ theta0``
     - 7
   * - dihedral–angle–angle
     - :math:`k\,\delta c_1\,\delta c_2\,P`
     - ``+ theta0_1, theta0_2``
     - 4

1.6 Reference offset
~~~~~~~~~~~~~~~~~~~~

.. math::

   E = E_0 \qquad \text{constant; zero force, zero virial}

A per-molecule shift that puts different molecular templates on a common
absolute energy scale. :math:`E_0` is the residual between the molecule's
reference atomization energy and the depth its Morse bonds already supply.


2. Short-range repulsion: tapered ZBL
-------------------------------------

Summed over every pair in the system. It is parameterized by atomic number
alone and has no free parameters.

.. math::

   a &= \frac{\text{SCREENING\_LENGTH}}{Z_1^{0.23} + Z_2^{0.23}}, \qquad
   x = r / a \\
   \phi(x) &= \sum_k C_k e^{-B_k x} \\
   f(r) &= \frac{1}{1 + \exp\!\left((r - r_\text{taper}) / w_\text{taper}\right)},
   \qquad \frac{df}{dr} = -\frac{f(1-f)}{w_\text{taper}} \\
   u(r) &= f(r)\, c_\text{ZBL}\, \frac{Z_1 Z_2\, \phi(x)}{r}

with

.. math::

   C &= (0.18175,\ 0.50986,\ 0.28022,\ 0.02817) \\
   B &= (3.19980,\ 0.94229,\ 0.40290,\ 0.20162) \\
   \text{SCREENING\_LENGTH} &= 0.46850\ \text{Å}

:math:`f` is a Fermi switch that turns the term **off** above
:math:`r_\text{taper}` (``taper_radius``, width ``taper_width``);
:math:`c_\text{ZBL}` is ``zbl_ccoul``.


3. Dispersion and contact: switched 12-6
----------------------------------------

Summed over every pair. Pair parameters are geometric means of the per-atom
ones, :math:`\sigma_{ij} = \sqrt{\sigma_i\sigma_j}` and
:math:`\varepsilon_{ij} = \sqrt{\varepsilon_i\varepsilon_j}`.

.. math::

   g(r) &= 1 - \frac{1}{1 + \exp\!\left((r - r_\text{switch}) / w_\text{switch}\right)},
   \qquad \frac{dg}{dr} = \frac{g(1-g)}{w_\text{switch}} \\
   s(r) &= c + (r/\sigma)^6, \qquad \frac{ds}{dr} = \frac{6 r^5}{\sigma^6} \\
   u_\text{12-6}(r) &= 4\varepsilon \left[ s^{-2} - s^{-1} \right], \qquad
   \frac{du_\text{12-6}}{dr} = 4\varepsilon \left[ s^{-2} - 2 s^{-3} \right] \frac{ds}{dr} \\
   u(r) &= g(r)\, u_\text{12-6}(r)

:math:`g` is the mirror image of the ZBL taper: it turns the term **on** above
:math:`r_\text{switch}` (``switch_radius``, width ``switch_width``). The soft core
:math:`c` (``soft_core``) makes :math:`u_\text{12-6}` finite at contact, where it
equals :math:`4\varepsilon(1/c^2 - 1/c)`. The minimum stays at :math:`s = 2`, so
the well depth is :math:`\varepsilon` for any :math:`c`, and :math:`c = 0` is the
bare 12-6. A pair with :math:`\sigma = 0` contributes nothing.

``switch_radius`` is deliberately **not** equal to ``taper_radius``. The two
switches are not complementary: between them is a gap where both terms are
small and electrostatics alone carries the hydrogen bond.


4. Electrostatics
-----------------

Electrostatics is evaluated per diabatic state and sits on the diagonal. The
global ``electrostatics`` parameter selects one of two terms:

- ``"acks2"``: fragment ACKS2, where each state solves for its own charges.
- ``"pointcharge"``: fixed charges carried by each template.

Point charges are fragment ACKS2's zero-softness limit.

4.1 Charge kernel
~~~~~~~~~~~~~~~~~

Both terms use the kernel of two Gaussian-smeared charges:

.. math::

   K_{ij} = \frac{\operatorname{erf}(\gamma r_{ij})}{r_{ij}}

It is finite at contact, where :math:`K \to 2\gamma/\sqrt\pi`, so charges saturate
rather than diverge as atoms approach.

Under open boundaries, the kernel uses the nearest image only and
:math:`K_{ii} = 0`. Under full periodicity it is the lattice sum, Ewald-split at
:math:`\kappa`:

.. math::

   K_{ij} &= \sum_{\mathbf{n}}{}' \frac{\operatorname{erf}(\gamma |\mathbf{r}_{ij} + \mathbf{n}|)
             - \operatorname{erf}(\kappa |\mathbf{r}_{ij} + \mathbf{n}|)}{|\mathbf{r}_{ij} + \mathbf{n}|}
          + \frac{4\pi}{V} \sum_{\mathbf{k} \ne 0}
             \frac{e^{-k^2 / 4\kappa^2}}{k^2} \cos(\mathbf{k} \cdot \mathbf{r}_{ij}) \\
          &\quad - \delta_{ij} \frac{2\kappa}{\sqrt\pi} + K_\text{bg},
          \qquad K_\text{bg} = -\frac{\pi}{\kappa^2 V}

Two properties apply only to the periodic form:

- :math:`K_{ii} \ne 0`. An atom interacts with its own periodic images, and this
  self-image term is added to the hardness rather than replaced by it.
- The :math:`\mathbf{k} = 0` term is omitted, and its neutralizing background
  :math:`K_\text{bg}` is put back explicitly. Omission alone is correct only for a
  charge-neutral contraction, since a constant added to every :math:`K_{ij}`
  contributes :math:`(\sum_i q_i)^2` to the energy. With the background, each
  individual :math:`K_{ij}` is well defined, which a charged cell, a point-charge
  exclusion and a difference between two states all need.

:math:`\kappa` and the reciprocal-space cutoff are both set by a single target
``accuracy``. Reciprocal vectors are selected on an integer ellipsoid rather than
by :math:`|\mathbf{k}|`, so the set is piecewise constant in the cell. A 3D
lattice sum is the right sum only under full periodicity, so a slab or wire falls
back to the nearest-image kernel.

4.2 Fragment ACKS2
~~~~~~~~~~~~~~~~~~

Per-atom parameters (the ``atom`` term): :math:`\mu` (electronegativity),
:math:`\eta` (hardness), ``soft_amp`` and ``soft_decay`` (the softness kernel),
and optionally :math:`q_0` (the reference charge, zero if absent).

Each state :math:`s` minimizes its own functional over the charges
:math:`\mathbf{q}` and Kohn–Sham potentials :math:`\mathbf{u}` of every atom in
the system:

.. math::

   F_s(\mathbf{q}, \mathbf{u}) = \boldsymbol\mu \cdot \mathbf{q}
     + \tfrac12 \mathbf{q}^\mathsf{T} (K + 2\,\mathrm{diag}\,\boldsymbol\eta)\, \mathbf{q}
     - \mathbf{u}^\mathsf{T} (\mathbf{q} - \mathbf{q}_{0,s})
     - \tfrac12 \mathbf{u}^\mathsf{T} L_{X_s} \mathbf{u}

State :math:`s` enters in two places:

- :math:`\mathbf{q}_{0,s}` are the reference charges of state :math:`s`'s
  templates. They carry each molecule's formal charge, so an H\ :sub:`3`\ O\ :sup:`+`
  holds its +1 and a proton hop moves it.
- The softness :math:`X_s` acts only between atoms in the same molecule of
  state :math:`s`:

  .. math::

     X_{ij} = \begin{cases}
       a_i a_j\, e^{-r_{ij}/\tau_{ij}}, \quad \tau_{ij} = \tfrac12 (d_i + d_j)
         & i, j \text{ in one molecule of } s \\
       0 & \text{otherwise}
     \end{cases}
     \qquad
     L_X = \mathrm{diag}\Big(\sum_j X_{ij}\Big) - X

  with :math:`a` = ``soft_amp`` and :math:`d` = ``soft_decay``.

The stationary point is the symmetric linear system
:math:`A_s \mathbf{x} = \mathbf{b}_s` in
:math:`\mathbf{x} = [\mathbf{q}, \mathbf{u}, \boldsymbol\lambda_q, \boldsymbol\lambda_u]`,
with one charge multiplier and one Kohn–Sham multiplier per molecule. Writing
:math:`M` for the (atoms × molecules) membership matrix of state :math:`s`:

.. math::

   A_s = \begin{pmatrix}
     K + 2\,\mathrm{diag}\,\boldsymbol\eta & -I & -M & 0 \\
     -I & -L_{X_s} & 0 & -M \\
     -M^\mathsf{T} & 0 & 0 & 0 \\
     0 & -M^\mathsf{T} & 0 & 0
   \end{pmatrix},
   \qquad
   \mathbf{b}_s = \begin{pmatrix}
     -\boldsymbol\mu \\ -\mathbf{q}_{0,s} \\ -M^\mathsf{T}\mathbf{q}_{0,s} \\ 0
   \end{pmatrix}

The third row holds every molecule at exactly its formal charge,
:math:`\sum_{i \in m} q_i = \sum_{i \in m} q_{0,i}`, so charge moves between
molecules only when the bonding changes. The kernel inside the solve is the full
kernel, with no exclusions. :math:`A` uses the bare kernel while the energy
carries ``ccoul``, so :math:`\mu` and :math:`\eta` are in the units that
convention implies rather than in eV directly.

**Energy.** The state's energy is its minimum, less each of its molecules'
isolated minimum. The isolated minimum is the same functional solved for that
molecule alone, under open boundaries:

.. math::

   E_\text{elec}(s) = c_\text{coul} \Big[ F_s^* - \sum_{m \in s} F_m^{*,\text{iso}} \Big],
   \qquad F^* = -\tfrac12\, \mathbf{b}^\mathsf{T} \mathbf{x}

A lone template therefore scores exactly zero, since its gas-phase energy
belongs to its bonded terms. What remains is the intermolecular electrostatics,
the polarization of each molecule by the others (including the intramolecular
half of that response), and every image interaction. No Coulomb exclusion is
needed or applied.

**Gradient.** Both :math:`F_s^*` and each :math:`F_m^{*,\text{iso}}` are
stationary in their own variables, so there is no charge-response term:

.. math::

   \frac{\partial E_\text{elec}}{\partial r} = \frac{c_\text{coul}}{2}
     \Big[ \mathbf{x}^\mathsf{T} \frac{\partial A_s}{\partial r} \mathbf{x}
       - \sum_m \mathbf{x}_m^{\text{iso}\,\mathsf{T}} \frac{\partial A_m}{\partial r} \mathbf{x}_m^\text{iso} \Big]

4.3 Point charges
~~~~~~~~~~~~~~~~~

Every template carries a ``charge`` term :math:`q` per atom, so the charges of
state :math:`s` are fixed by its bonding:

.. math::

   E_\text{elec}(s) = \frac{c_\text{coul}}{2} \sum_{ij} q_{s,i}\, q_{s,j}\, K_{ij}
     - c_\text{coul} \sum_{(i,j) \in \text{excl}(s)} q_{s,i}\, q_{s,j}\, K^\text{direct}_{ij}

The exclusion (§5.2) removes a molecule's own pairs from its bonded terms.
There is no linear solve, and no response term.

4.4 Coupling between blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The electrostatic energy is not local to an EVB block: every block's charges act
on every other block's atoms. Each block :math:`B` therefore sees the others
through their ground-state-averaged charges (a Hartree product):

.. math::

   \bar{\mathbf{q}}_B = \sum_s w_s\, \mathbf{q}_s, \qquad w_s = c_s^2

Here :math:`c_s` is the ground-state eigenvector component of state :math:`s`
in block :math:`B`. The total energy is :math:`\sum_B \lambda_B` minus the
block–block energy of the averaged charges, which that sum counts twice. The
blocks are swept in turn until no weight moves. With at most one multi-state
block, the first sweep is already converged. Under fragment ACKS2, the atoms
outside every multi-state block respond to each state exactly, not through a
mean.

A reaction never changes the total charge, so under full periodicity a charged
cell's uniform background contributes the same energy to every state.


5. Exclusions
-------------

The repulsion sums of §2 and §3 run over every pair with no reference to the bond
graph. Pairs separated by ``exclusion_depth`` bonds or fewer are then removed, so
that a molecule's internal geometry is set by its bonded terms. This subtraction
depends on the bond graph and is evaluated per state as ordinary additive terms.

5.1 Repulsion and dispersion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   E_\text{excl,12-6} &= -u_\text{12-6}(r; \sigma_{ij}, \varepsilon_{ij}) \\
   E_\text{excl,ZBL} &= -u_\text{ZBL}(r; Z_1, Z_2)

These use the same functional forms as §2 and §3, switches included, so the
cancellation is exact. A 12-6 exclusion is present only for pairs whose atoms
both carry nonzero :math:`\sigma` and :math:`\varepsilon`. Where either is zero,
the whole-system 12-6 is identically zero for that pair and there is nothing to
cancel. ZBL has no such exemption, since it has no free parameters.

5.2 Coulomb exclusion (point charges only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``exclude_coulomb`` set, each excluded pair also carries a
``coulombexclusion`` term. Only the point-charge term reads it (§4.3). Fragment
ACKS2 subtracts each molecule's isolated minimum instead and ignores these terms.

The exclusion removes the direct pair and not its images. It always uses the
open-boundary kernel :math:`K^\text{direct}_{ij} = \operatorname{erf}(\gamma r)/r`
at the minimum-image separation. Under Ewald, :math:`K_{ij}` also contains
:math:`i`'s interaction with every image of :math:`j`, and that part does not
belong to the bonded terms.


6. Coupling: the off-diagonal
-----------------------------

One Gaussian per reaction channel, in Å and eV. There are three forms, and the
channel's own connectivity change selects which one applies:

- A **fission** breaks or forms one bond, separating two fragments.
- A **transfer** breaks one bond and forms another, the two sharing an atom.
- Any other change uses the **RMSD** fallback.

.. list-table::
   :header-rows: 1

   * - form
     - channel
     - :math:`V`
     - parameters
   * - two-body
     - fission
     - :math:`A\, e^{-a (r - r_0)^2}`
     - ``A, a, r0``
   * - three-body
     - transfer
     - :math:`A\, e^{-a g}`
     - ``A, a, ra0, rb0, t0``
   * - RMSD
     - anything else
     - :math:`\frac{A}{M} \sum_m e^{-a \rho_m^2}`
     - ``A, a`` + a TS ensemble

**Two-body.** :math:`r` is the length of the single bond that changes.

**Three-body.** On the ordered triple of donor :math:`D`, transferring atom
:math:`H` and acceptor :math:`X`, with the mover central:

.. math::

   r_a &= |\mathbf{r}_H - \mathbf{r}_D|, \quad
   r_b = |\mathbf{r}_H - \mathbf{r}_X|, \quad
   d = |\mathbf{r}_X - \mathbf{r}_D| \\
   d_0 &= \sqrt{r_{a0}^2 + r_{b0}^2 - 2 r_{a0} r_{b0} \cos t_0} \\
   g &= (r_a - r_{a0})^2 + (r_b - r_{b0})^2 + (d - d_0)^2

:math:`t_0` is stored as an angle because that is the readable parameter. It is
converted to a length so that one width :math:`a` is dimensionally consistent
across all three sides. The three sides describe the triangle completely and
without redundancy. At the reference triangle :math:`g = 0`, so :math:`V = A`
exactly.

**RMSD.** :math:`\rho_m` is the optimally superposed RMSD to the :math:`m`-th of
the channel's :math:`M` stored transition-state frames: rotation and translation
are removed, weights are unity, and nothing is rescaled. An RMSD to a fixed
template does not change under rotation or translation but does change under
scaling, so this form has a nonzero virial. Under periodic boundaries, the
fragment is unwrapped by cumulative minimum image before superposition.

6.1 Where the amplitude comes from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - form
     - amplitude from
     - width measured in
   * - two-body
     - the diabatic crossing
     - the breaking bond's length
   * - three-body
     - the reference barrier
     - the transferring atom's triangle
   * - RMSD
     - the reference barrier
     - the RMSD over all 3N coordinates

For a channel with a true saddle, the amplitude is the reference barrier
:math:`E^*` inverted through the 2×2 secular equation at the transition state,
where :math:`V = A` exactly:

.. math::

   A = -\sqrt{(H_m - E^*)^2 - \Delta H^2}, \qquad
   H_m = \tfrac12 (H_R + H_P), \quad \Delta H = \tfrac12 (H_R - H_P)

A fission has no saddle, so it has no barrier to invert. The bound diabat *is*
the ground state up to the crossing, so inverting a barrier would give
:math:`A = 0`, a channel that never opens. A fission coupling is therefore
centred on the diabatic crossing instead. At the crossing the diabats are
degenerate, so the stabilization :math:`\operatorname{hypot}(\Delta H, V) - |\Delta H|`
equals :math:`|A|` exactly, and the Gaussian is at its maximum exactly where the
topology decision is taken.

A transfer's two diabats need not cross along the transfer coordinate at all,
so a crossing-centred fit is wrong for it.

6.2 Why a transfer does not use an RMSD width
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An RMSD is a tolerance on all :math:`3N` coordinates at once, so thermal motion
of any spectator atom switches the coupling off. The triangle form depends only
on the three atoms that take part in the transfer. The RMSD form remains only
as the fallback for a channel that is neither a fission nor a transfer. For
example, two independent bond changes that share no atom have no single
coordinate for the coupling to depend on.


7. Parameters
-------------

7.1 Per-term parameters
~~~~~~~~~~~~~~~~~~~~~~~

As held in memory and as stored in a ``.jsonl``.

.. list-table::
   :header-rows: 1

   * - term
     - parameters
     - units
   * - bond (Morse)
     - ``D, r0, k, h``
     - eV, Å, eV/Å², eV
   * - angle
     - ``theta0, k``
     - rad, eV
   * - bond–bond
     - ``r1_0, r2_0, k``
     - Å, Å, eV/Å²
   * - bond–angle
     - ``theta0, r0, k``
     - rad, Å, eV/Å
   * - angle–angle
     - ``theta1_0, theta2_0, k``
     - rad, rad, eV
   * - periodic dihedral
     - ``phi0, n, k``
     - rad, —, eV
   * - dihedral–bond
     - ``+ r0``
     - Å
   * - dihedral–angle
     - ``+ theta0``
     - rad
   * - dihedral–angle–angle
     - ``+ theta0_1, theta0_2``
     - rad
   * - reference
     - ``E0``
     - eV
   * - ACKS2, per atom (``atom``)
     - ``mu, eta, soft_amp, soft_decay, q0``
     - see §4.2; ``q0`` in e
   * - point charge, per atom (``charge``)
     - ``q``
     - e
   * - 12-6, per atom
     - ``sigma, eps``
     - Å, eV
   * - ZBL
     - none (atomic numbers only)
     - —
   * - two-body coupling
     - ``A, a, r0``
     - eV, 1/Å², Å
   * - three-body coupling
     - ``A, a, ra0, rb0, t0``
     - eV, 1/Å², Å, Å, rad
   * - RMSD coupling
     - ``A, a`` + TS ensemble
     - eV, 1/Å²

Exclusion terms have no parameters of their own: they reuse the 12-6 pair
parameters, the atomic numbers and the charge kernel.

7.2 Global parameters
~~~~~~~~~~~~~~~~~~~~~

These belong to a *dataset*, not to the force field: a parameter set is fitted at
particular values and is valid only at those values. A dataset states them in
its manifest's ``global_params`` block
(:class:`~DynamicTopology.forcefield.params.ForceFieldParams`), and the defaults
below apply wherever it does not.

.. list-table::
   :header-rows: 1

   * - parameter
     - default
     - unit
     - enters
   * - ``bond_asymptote``
     - 1.0
     - eV
     - Morse :math:`h` fallback (§1.1)
   * - ``taper_radius``
     - 1.5
     - Å
     - ZBL switch (§2)
   * - ``taper_width``
     - 0.12
     - Å
     - ZBL switch (§2); the default ``switch_width``
   * - ``switch_radius``
     - 2.2
     - Å
     - 12-6 switch (§3)
   * - ``switch_width``
     - ``taper_width``
     - Å
     - 12-6 switch (§3)
   * - ``soft_core``
     - 0.01
     - —
     - 12-6 soft core (§3)
   * - ``exclusion_depth``
     - 3
     - bonds
     - exclusions (§5)
   * - ``exclude_coulomb``
     - true
     - —
     - point-charge exclusions (§5.2)
   * - ``electrostatics``
     - ``"acks2"``
     - —
     - fragment ACKS2 or point charges (§4)
   * - ``gamma``
     - 2.0
     - 1/Å
     - charge kernel (§4.1)
   * - ``accuracy``
     - 1e-8
     - —
     - Ewald splitting and cutoff (§4.1)
   * - ``ccoul``
     - 14.4
     - eV·Å
     - electrostatic energy (§4)
   * - ``zbl_ccoul``
     - 14.399645
     - eV·Å
     - ZBL (§2)

``ccoul`` and ``zbl_ccoul`` are the same physical constant to different
precision. They are separate fields so that unifying them cannot silently move
one of the two terms for a parameter set that is already fitted.

Every parameter except ``accuracy`` enters :math:`E_\text{bonded} + E_\text{nonbonded}`,
which the templates are fitted against, so changing one invalidates that
dataset's fitted parameters.
