# DynamicTopology System Equation Reference

Method and equation reference for the reactive `DynamicTopology` ASE calculator
([ase.py:13](ase.py#L13) → [system.py:103](system.py#L103)).

Every force evaluation does five things at the current nuclear geometry $\mathbf{x}$:

1. solve the **charge equilibration** once, on the whole system, from the unmasked kernel;
2. partition the system into independent **blocks** of coupled reaction channels;
3. close a **diabatic basis** $\{|i\rangle\}$ over each block by a geometric admission criterion;
4. build and diagonalize a small **EVB Hamiltonian** per block, taking the ground root;
5. add the three **whole-system nonbonded** sums on top, the electrostatic one screened by
   what the ground state has just decided is bonded.

Energy, forces **and virial** are carried through all five steps together; the
virial is contracted with the same ground-state eigenvector as the forces and
published as `stress` by the calculator (§2.4).

The eigenvector of the ground root also selects the bond topology carried into
the next MD step — that feedback is the "dynamic topology".

### Notation and units

| Symbol | Meaning |
| --- | --- |
| $\mathbf{x}$ | nuclear coordinates (Å) |
| $T$ | a *topology*: a bond graph over the atoms, with parameter terms attached |
| $H_{ii}$ | diabatic (state) energy of topology $i$ |
| $V_{ij}$ | off-diagonal coupling between topologies $i$ and $j$ |
| $c_i$ | ground-state eigenvector amplitude on state $i$; $c_i^2$ is its weight |
| $u(r)$ | 12-6 pair potential |
| $\rho$ | RMSD to a stored transition-state reference geometry |
| $\mathbf{W}$ | virial, $W_{ab} = \partial E/\partial \epsilon_{ab}$ under a homogeneous strain |
| $K_{ij}$ | the ACKS2 charge kernel, $\operatorname{erf}(\gamma r)/r$ or its lattice sum (§2.3.1) |
| $S_{ij}$ | the intramolecular exclusion screen, $1 - \sum_s w_s M_s$ (§2.3.2) |
| $C(T_i)$ | the Coulomb exclusion correction on state $i$'s diagonal (§2.3.2) |

External units are ASE's: **eV** and **Å**. `QForce` and `LennardJones` work
internally in q-force's **kJ/mol** and **nm** and convert at the end of
`__call__`; the per-term `compute_*` methods return *unconverted* values.
Forces are $\mathbf{F} = -\partial E/\partial \mathbf{x}$ throughout, and every
force field's `__call__` returns the 3-tuple $(E, \mathbf{F}, \mathbf{W})$.

**None of the pair sums is truncated.** `ZBL`, `LennardJones` and `ACKS2` are each a
dense $N\times N$ evaluation — no neighbour list, no cutoff radius. That is
affordable at the box sizes here, and it is the first thing to revisit if the system
size grows.

The two short-ranged terms stop at the nearest image. **Electrostatics does not**,
because it is not short ranged: under full periodicity the charge kernel is summed
over every image (§2.3.1), and `acks2.py` never branches on `pbc` — the kernel is an
object with two implementations behind one interface.

**Each of the three sums is still exactly the same number on every diabatic state**
(§4.2), and that is what keeps all three outside the Hamiltonian. What is *not* the
same on every state is which pairs are intramolecular, and that correction does sit
on the diagonal — §2.3.2, which is also where the one term that cannot be written
additively is dealt with.

### Global parameters belong to the dataset

Thirteen constants appear in the equations below with no symbol of their own — two
taper radii and their widths, the Morse asymptote, the exclusion depth, the charge
smearing width, the Coulomb constant. They are **not module constants**. Each is a
field of `params.ForceFieldParams` ([forcefield/params.py](forcefield/params.py)),
stated per dataset under `global_params` in the manifest, defaulted when the
manifest omits it, and read through `params.active()` at call time:

```json
"global_params": { "taper_radius": 1.5, "exclusion_depth": 3 }
```

All but `accuracy` and `shape_decay` sit inside $E_{\text{bonded}} +
E_{\text{nonbonded}}$, which `fit/dissociation.py` solves each template's depth
scale against (§5.2), so each one is part of the definition of the surface a
dataset's `.jsonl` files were fitted to. As module constants they were a property of
the installed source tree instead, and a checkout whose taper radius had moved
scored every dataset on a surface it was not fitted to and said nothing about it.
`forcefield/README.md` tabulates the full set with units and defaults.

`ReactionSet.load` reads the block before it touches a template — the exclusion
derivation (§2.3.2) and the reference shift (§5.1) both depend on it — and activates
it process-wide. Two datasets whose blocks disagree therefore cannot be loaded into
one process: `params.activate` refuses the second, naming the fields, because
whichever loaded second would score the first's templates on the wrong surface.
`params.use(...)` is the explicit override for a caller that wants a different
surface on purpose.

---

## 1. Topology, reactions, and state generation

### 1.1 Topology identity

A topology is an `nx.Graph` of bonds. Its identity is the Weisfeiler–Lehman
graph hash over `atomic_number`,

$$
h(T) \;=\; \mathrm{WL}\big(G_T,\ \text{atomic\_number}\big),
$$

which keys the molecule and reaction databases in `ReactionSet`. Bond
perception, where it is used, admits a bond when

$$
r_{ij} \;<\; 1.3\,\big(R^{\text{cov}}_i + R^{\text{cov}}_j\big),
$$

with the scale factor `BOND_SCALE = 1.3` ([core/topology.py:23](core/topology.py#L23)).

Connected components of $G_T$ are the *molecules*; subgraphs keep global node
indices, which is what makes template remapping work everywhere else.

### 1.2 Applicable reactions

A stored `Reaction` is a triple (reactant template $R$, product template $P$,
ensemble of transition-state geometries $\{\mathbf{y}^{(m)}\}$).

A reaction applies to a live topology $T$ through an index map
$\pi: V(R) \to V(T)$ found by subgraph isomorphism with element matching:

$$
\pi \in \mathrm{Iso}\big(R,\ T\big), \qquad Z_{\pi(v)} = Z_v .
$$

All distinct isomorphisms are enumerated (`get_mappings`), because
symmetry-equivalent maps address different atoms and therefore carry different
couplings.

The bonds a channel changes are a pure function of the templates and $\pi$
(`Reaction.apply`, [core/reaction.py:109](core/reaction.py#L109)):

$$
\mathcal{B} = E\big(\pi R\big)\setminus E\big(\pi P\big), \qquad
\mathcal{F} = E\big(\pi P\big)\setminus E\big(\pi R\big),
$$

and applying it is the edge rewrite $E(T') = \big(E(T)\setminus\mathcal{B}\big)\cup\mathcal{F}$.

**Network construction** (`ReactionSet.get_network`, [core/reactionset.py:517](core/reactionset.py#L517)).
Nodes are molecules. Unimolecular channels are self-loops. A bimolecular channel
between molecules $\mu,\nu$ is offered only when their minimum-image separation
is under the cutoff:

$$
\min_{i\in\mu,\ j\in\nu} \; \big\| \mathbf{x}_{ij} - \mathbf{L}\,\mathrm{round}\!\left(\mathbf{L}^{-1}\mathbf{x}_{ij}\right) \big\|
\;\le\; r_{\text{bimol}} \quad (\text{default } 4\ \text{Å}),
$$

with $\mathbf{L}$ the cell matrix, applied per periodic direction.

### 1.3 Blocks: which channels share a Hamiltonian

Channels are grouped by **shared atoms**, not by molecular proximity
([core/network.py:51](core/network.py#L51)). Let $A_r = \mathrm{im}\,\pi_r$ be the
atom set a channel touches; the block relation is the transitive closure of

$$
r \sim s \iff A_r \cap A_s \neq \emptyset .
$$

Atom-disjoint reactions commute — the state in which both occurred is the
product of the two single-reaction states — so their blocks factorize and their
energies add:

$$
E \;=\; \sum_{B} \varepsilon_0^{(B)} \;+\; E_{\text{nonbonded}} .
$$

Putting unrelated channels in one matrix is not size consistent: a star of $N$
degenerate leaves is stabilized by $\sqrt{N}\,|V|$, so the ground state would
depend on how many irrelevant channels happen to be nearby.

### 1.4 Basis closure (state generation)

The basis must be a function of $\mathbf{x}$ alone. "Current topology plus
everything one reaction away" is not — for three or more states the reachable set
depends on the seed. Instead the basis is closed under a **geometric admission
criterion** (`_switch`, [basis.py:501](basis.py#L501); `_close`, [basis.py:559](basis.py#L559)).

For a candidate edge parent $\to$ child with coupling $V$, define the half gap
and the two-level stabilization

$$
\eta \;=\; \tfrac{1}{2}\big(H_{\text{child}} - H_{\text{parent}}\big),
\qquad
\Delta_{\text{stab}} \;=\; \sqrt{\eta^2 + V^2} \;-\; |\eta| ,
$$

which is exactly how far the lower root of the corresponding $2\times2$ block
lies below $\min(H_{\text{parent}}, H_{\text{child}})$. The channel is admitted
when $\Delta_{\text{stab}} > \epsilon$.

Testing $|V| > \epsilon$ instead would be wrong in both directions:
$\Delta_{\text{stab}}$ is *quadratic* in $V$ for well-separated diabats and
*linear* for degenerate ones.

**Ramp, not step.** Since dropping a state costs precisely $\Delta_{\text{stab}}$,
a hard threshold moves the energy discontinuously by up to $\epsilon$. The
coupling is therefore scaled by a quintic switching weight

$$
w \;=\;
\begin{cases}
0, & \Delta_{\text{stab}} \le \epsilon \\[2pt]
S\!\left(\dfrac{\Delta_{\text{stab}} - \epsilon}{\delta}\right), & \epsilon < \Delta_{\text{stab}} < \epsilon + \delta \\[8pt]
1, & \Delta_{\text{stab}} \ge \epsilon + \delta
\end{cases}
\qquad
S(t) = t^3\big(6t^2 - 15t + 10\big),
$$

with $\epsilon = $ `eps` $= 10^{-3}$ eV and $\delta = $ `switch_width`, which
defaults to `eps` ([basis.py:207](basis.py#L207), [basis.py:220](basis.py#L220)) —
so a channel is decoupled at $\Delta_{\text{stab}} = \epsilon$ and at full strength
at $2\epsilon$. $S$ is $C^2$ with $S'(0)=S'(1)=0$ (`_smoothstep`,
[basis.py:108](basis.py#L108); slope $S'(t) = 30t^2(t-1)^2$,
[basis.py:118](basis.py#L118)), so a state joins the basis decoupled and gains its
coupling smoothly. $\delta = 0$ recovers the step exactly. Because the ramp runs
*upward* from $\epsilon$, the admitted set is exactly what the hard gate admitted;
the switch changes the energy, never the membership.

The effective off-diagonal is $V^{\text{eff}} = w\,V$, and its gradient carries
both factors:

$$
\frac{\partial V^{\text{eff}}}{\partial \mathbf{x}}
= w\,\frac{\partial V}{\partial \mathbf{x}} + V\,\frac{\partial w}{\partial \mathbf{x}},
\qquad
\frac{\partial w}{\partial \mathbf{x}}
= \frac{S'(t)}{\delta}\left[
\left(\frac{\eta}{\sqrt{\eta^2+V^2}} - \operatorname{sgn}\eta\right)\frac{\partial \eta}{\partial \mathbf{x}}
+ \frac{V}{\sqrt{\eta^2+V^2}}\frac{\partial V}{\partial \mathbf{x}}
\right].
$$

Dropping the second term leaves forces inconsistent with the energy exactly in
the transition-state region. `tests/test_gradients.py::TestSystemGradients` pins
this: stubbing out $\partial w/\partial\mathbf{x}$ gives a finite-difference error
of 6.0e-03 where the correct forces agree to 1.7e-08.

The **same chain rule runs in strain** ([basis.py:402](basis.py#L402)), with the
force-convention sign flips absent because the block stores virials
($+\partial E/\partial\epsilon$) rather than forces:

$$
\frac{\partial w}{\partial \boldsymbol{\epsilon}}
= \frac{S'(t)}{\delta}\left[
\left(\frac{\eta}{\sqrt{\eta^2+V^2}} - \operatorname{sgn}\eta\right)\frac{\partial \eta}{\partial \boldsymbol{\epsilon}}
+ \frac{V}{\sqrt{\eta^2+V^2}}\frac{\partial V}{\partial \boldsymbol{\epsilon}}
\right],
\qquad
\frac{\partial V^{\text{eff}}}{\partial \boldsymbol{\epsilon}}
= w\,\frac{\partial V}{\partial \boldsymbol{\epsilon}} + V\,\frac{\partial w}{\partial \boldsymbol{\epsilon}}.
$$

The weight itself is computed only when it can matter: `_channel_weight` returns
early with no gradients when $w \le 0$ or $w \ge 1$ ([basis.py:380](basis.py#L380)),
since both ends of the ramp are flat.

**Screening.** The gate needs only the *gap*, and every molecule the reaction
leaves alone contributes equally to both diabats, so the test is evaluated on the
reacting fragment only — an exact cancellation, not an approximation
(`_channel_weight`, [basis.py:340](basis.py#L340); `_local_energy`,
[basis.py:302](basis.py#L302)). The gate runs on subgraphs of the *parent*, with
the reaction's broken and formed edges applied in place
([basis.py:371](basis.py#L371)) — the product topology is built only after the
channel passes ([basis.py:611](basis.py#L611)). Only admitted states are then
evaluated in full. Four caches (`_energy_cache`, `_reaction_cache`,
`_coupling_cache`, `_molecule_cache`) are cleared at the top of every `build`
([basis.py:537](basis.py#L537)), so they are per-geometry only.

**Seed invariance.** The criterion is symmetric under exchanging parent and child
($\eta \to -\eta$ leaves $\eta^2$ and $|\eta|$ fixed), so the admitted set is the
connected component of the gate-passing state graph containing the seed — and
every member of that component generates the same component. States are then
sorted into a canonical order, so the block does not remember what seeded it.
Truncation (`max_states = 64`, `max_depth`) is the one thing that breaks the
invariance. `max_depth` defaults to `max_states` ([basis.py:228](basis.py#L228))
because a small depth limit reintroduces seed dependence directly — at 4 it cut
the 7-state H₂O + HO basis to 5 states from some seeds and 7 from others.
Expansion continues past the cap rather than stopping, so a `Block` reports
`capped` exactly when a gate-passing state was actually refused
([basis.py:630](basis.py#L630)), and logs a warning.

---

## 2. Component force fields

### 2.1 Bonded terms (`QForce`, [forcefield/qforce.py](forcefield/qforce.py))

Dispatch is by convention: `__call__` iterates the term dictionary and calls
`compute_<type>`, skipping types with no method. Each returns the 3-tuple
$(E, \mathbf{F}, \mathbf{W})$ in q-force's internal units; `__call__` converts
([qforce.py:69](forcefield/qforce.py#L69)). `bond_form` is validated in
`__init__` and defaults to `"morse"`.

**Bond — Morse with a one-sided Hulburt–Hirschfelder shape term** (default;
required for reactive work, `compute_bond` [qforce.py:135](forcefield/qforce.py#L135)
→ `_bond_morse` [qforce.py:140](forcefield/qforce.py#L140)):

$$
E = D\Big[\big(1 - e^{-\alpha \Delta r}\big)^2 - 1 + c\,s^3 e^{-b s}\Big],
\qquad
\alpha = \sqrt{\frac{k}{2D}},\quad
\Delta r = r - r_0,\quad
s = \alpha\,\max(\Delta r, 0),
$$

with $b$ fitted per bond type alongside $c$; `shape_decay` $= 4.0$
([params.py](forcefield/params.py)) is only the fallback a term file
written before $b$ was a parameter reads as. The $-D$ offset
puts the dissociated limit at zero, so a topology's energy carries the depth of
the bonds it contains and breaking a bond costs $+D$; boundedness above is what
keeps a product diabat whose new bond is still several Å long from costing an
unbounded $\tfrac12 k \Delta r^2$.

The shape term is $O(s^3)$, so it perturbs neither $D$, $r_0$, nor the curvature
at $r_0$, and it decays to zero, so the dissociation limit is untouched. It
exists because plain Morse ($c=0$) is exact at the minimum and at dissociation
with nothing left over in between, and came out 0.55–1.83 eV too deep at the
stretched geometries where reactions happen. Removing it costs every coupling:
with $c=0$ none of the thirteen metathesis channels can be inverted. Buying them
back by inflating $k$ instead does not work either — reaching even 17 of 19
needed H₂ at **12402 cm⁻¹** against an experimental 4401, and no force constant
reached 18.

The relevant $s$-bands, which are what fixes the useful range of $b$ (the term
peaks at $s = 3/b$): bonds at their own equilibrium geometry sit at
$s = 0.180$–$0.382$; metathesis transition states at $0.352$–$1.049$ (median
$0.681$); homolysis mid-dissociation at $1.081$–$1.525$ (median $1.358$). Sweeping
$b$ with everything else refitted ([qforce.py:26](forcefield/qforce.py#L26)):

| $b$ | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 | 6.0 |
| --- | --- | --- | --- | --- | --- | --- |
| $c_\mathrm{max}$ | 1.31 | 3.84 | 7.80 | 19.33 | 34.50 | 52.20 |
| metathesis channels | 12/13 | 13/13 | 13/13 | 13/13 | 13/13 | 12/13 |
| dissociation rms (eV) | 0.700 | 0.783 | 1.098 | 1.710 | 2.660 | 3.181 |
| fastest mode (cm⁻¹) | 4517 | 4402 | 4352 | 4400 | 4402 | 4432 |
| stable $dt$ (fs) | 0.492 | 0.505 | 0.511 | 0.505 | 0.505 | 0.502 |

At $b = 2.0$ the $c_\mathrm{max}$ bound binds on five of eight bonds, which is why
the fit searches $c/c_\mathrm{max}(b)$ with $b$ free rather than pinning either.

That $O(s^3)$ property is about $r_0$ and **not** about the vibrational
frequency, which is a distinction this section used to elide. `fit_bond_lengths`
displaces $r_0$ inside the bond so the Morse can lean against the repulsion, and
at that displacement the term's curvature,
$D c \alpha^2 s\,(b^2s^2 - 6bs + 6)\,e^{-bs}$, is not small. Its bracket is
negative for $1.268/b < s < 4.732/b$, so whether the shape term stiffens or
softens a bond depends on where that bond sits relative to $3/b$ — which is one
of the reasons $b$ is fitted rather than fixed.

$c$ is bounded by the requirement that the curve still dissociate downhill, and
that bound moves steeply with $b$ — $1.31$ at $b=2$, $19.33$ at $b=4$, $52.20$
at $b=6$ — so the fit searches the fraction $c/c_\mathrm{max}(b)$ rather than
$c$ itself. See `fit.dissociation.shape_bound`.
It is clamped at $\Delta r = 0$: continued to compression, $s^3 e^{-bs}$ would
outgrow the $e^{-2\alpha\Delta r}$ wall and run to $-\infty$. The clamp is $C^2$
because the term is cubic there.

Radial derivative:

$$
\frac{dE}{dr} = 2D\big(1-e^{-\alpha\Delta r}\big)\alpha e^{-\alpha\Delta r}
+ D\,c\,\alpha\,s^2\big(3 - b s\big)e^{-bs}.
$$

**Bond — harmonic** (non-reactive use only, [qforce.py:218](forcefield/qforce.py#L218)):
$E = \tfrac12 k \Delta r^2$; $D$ is unused.

**Reference shift** ([qforce.py:236](forcefield/qforce.py#L236)):
$E = \sum_{\text{mol}} E_0$, geometry-independent, hence zero force. Diabatic
states are compared by *absolute* energy, so every bonding topology must be
measured from the same zero; see §5.1 for how $E_0$ is set.

**Angle** (cosine-harmonic):

$$
E = k\big(\cos\theta - \cos\theta_0\big)^2,
\qquad
\cos\theta = \hat{\mathbf{v}}_a\!\cdot\!\hat{\mathbf{v}}_b .
$$

**Cross terms** (q-force's coupling terms). Two of the three carry a clip that
bounds a pathological attractive excursion; `angleangle` carries **none**, because
both of its factors are bounded by $\pm 2$ already:

$$
\begin{aligned}
E_{\text{bond-bond}} &= \max\Big(k\,\Delta r_1 \Delta r_2,\ -10\Big) \\
E_{\text{bond-angle}} &= \max\Big(k\,\Delta r\,\big(\cos\theta - \cos\theta_0\big),\ -20\Big) \\
E_{\text{angle-angle}} &= k\big(\cos\theta_1 - \cos\theta_{1,0}\big)\big(\cos\theta_2 - \cos\theta_{2,0}\big)
\end{aligned}
$$

Where a clip is active the gradient is masked off with it
([qforce.py:343](forcefield/qforce.py#L343),
[qforce.py:380](forcefield/qforce.py#L380)), so the clipped branch is flat rather
than kinked in the derivative as well as the energy.

**Dihedrals.** With $\varphi = \operatorname{atan2}(S, C)$ built from the
components of $\mathbf{v}_a$ and $\mathbf{v}_c$ perpendicular to the central bond
axis $\hat{\mathbf{n}} = \mathbf{v}_b/\|\mathbf{v}_b\|$,

$$
\mathbf{u} = P\mathbf{v}_a,\quad \mathbf{v} = P\mathbf{v}_c,\quad P = I - \hat{\mathbf{n}}\hat{\mathbf{n}}^{\mathsf T},
\qquad
S = (\hat{\mathbf{n}}\times\mathbf{u})\cdot\mathbf{v},\quad C = \mathbf{u}\cdot\mathbf{v},
$$

$$
\begin{aligned}
E_{\text{periodic}} &= k\big[1 + \cos(n\varphi - \varphi_0)\big] \\
E_{\text{dih-bond}} &= k\,\Delta r\big[1 + \cos(n\varphi - \varphi_0)\big] \\
E_{\text{dih-angle}} &= k\big(\cos\theta - \cos\theta_0\big)\big[1 + \cos(n\varphi - \varphi_0)\big] \\
E_{\text{dih-angle-angle}} &= k\,\Delta\cos\theta_1\,\Delta\cos\theta_2\big[1 + \cos(n\varphi - \varphi_0)\big]
\end{aligned}
$$

with analytic $\partial\varphi/\partial\mathbf{x}$ given in
`_dihedral_phi_and_grads` ([qforce.py:456](forcefield/qforce.py#L456)), which
floors $S^2 + C^2$ at $10^{-30}$ so a collinear frame yields zero rather than a
division by zero.

Every analytic gradient above is checked against central finite differences in
`tests/test_gradients.py` and every virial against strained-cell differences in
`tests/test_stress.py`; a new `compute_*` method must get a case in **both**.

### 2.2 Repulsion (`ZBL`, [forcefield/zbl.py](forcefield/zbl.py), and §2.2.1)

**What it is for.** `ACKS2` has no repulsive branch. Its kernel $\operatorname{erf}(2r)/r$ is
*finite* at contact rather than divergent (it tends to $4/\sqrt{\pi} = 2.257$), so the
charges saturate and the solve does not run away — but the interaction between two
atoms of different electronegativity is a smooth, monotone $\sim 4$ eV attractive
funnel all the way to zero separation, with no minimum and no wall:

| $r$ (Å) | 2.0 | 1.5 | 1.2 | 0.96 | 0.6 | 0.2 | 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $Q_{\mathrm H}$ (e) | 0.110 | 0.212 | 0.259 | 0.286 | 0.317 | 0.347 | 0.353 |
| $E$ (eV) | −0.09 | −0.43 | −0.80 | −1.21 | −2.19 | −3.71 | **−4.03** |

At 3000 K, $k_BT = 0.26$ eV. A 200-atom H₂/O₂ box run against `ACKS2` alone reached
0.60 Å intermolecular contacts with 55 pairs inside 1.2 Å, reading −46.6 eV of
nonbonded energy — 0.85 eV per pair, which is this table.

**The form.** The ZBL universal screened-nuclear potential, in Å and eV:

$$
u(r) = \frac{k\,Z_iZ_j}{r}\,\varphi\!\left(\frac{r}{a}\right),
\qquad
a = \frac{0.4685}{Z_i^{0.23} + Z_j^{0.23}},
\qquad
k = 14.399645,
$$

$$
\varphi(x) = 0.18175\,e^{-3.1998x} + 0.50986\,e^{-0.94229x} + 0.28022\,e^{-0.4029x} + 0.02817\,e^{-0.20162x}.
$$

The constants are `SCREENING_LENGTH` $= 0.46850$ Å ($= 0.8854\,a_0$) and the
amplitude/decay tuples `PHI_C` / `PHI_B` ([zbl.py](forcefield/zbl.py)), plus
`zbl_ccoul` $= 14.399645$ eV·Å from the global parameters. Note `zbl_ccoul` and
`ccoul` $= 14.4$ are the same physical constant written to different precision,
which is why they are two fields of `ForceFieldParams` and not one.

**The taper.** ZBL is a screened *nuclear* potential, fitted where two nuclei are
close enough that the electrons between them barely intervene. It carries no
information about the range where chemistry happens, and used unmodified it reaches
straight into the hydrogen bond — +0.55 eV for O···H at 1.94 Å, against a hydrogen
bond worth −0.218 eV. So it is switched off with a Fermi function, as every hybrid
ZBL potential in the literature does (Tersoff/ZBL, ReaxFF):

$$
u(r) \;\longrightarrow\; f(r)\,u(r),
\qquad
f(r) = \frac{1}{1 + e^{(r - r_c)/w}},
\qquad
r_c = 1.5\ \text{Å},\quad w = 0.12\ \text{Å}.
$$

(`taper_radius` and `taper_width`, [params.py](forcefield/params.py).) The
taper is folded in **inside** `pair_potential` rather than applied by the caller
([zbl.py:269](forcefield/zbl.py#L269)), so $f$ and $f'$ can never travel
separately — `fit/dissociation.py` differentiates this function for its bond
curvatures and would otherwise see an inconsistent pair. The Fermi exponent is
clipped to $\pm 500$ ([zbl.py:225](forcefield/zbl.py#L225)): the argument already
reaches 54 at 8 Å, and unclipped a half-cell-apart pair produces `0 * inf = nan`
in the derivative rather than the zero it should give.

Those potentials taper at $\sim 1$ Å and hand over to their own repulsion; here
there is nothing to hand over to, so $r_c$ sits as far out as the wall can afford.
It is bounded from below by the `ACKS2` contact funnel (at $r_c = 1.0$ Å the H₂ + O₂
approach of `tests/test_collapse.py` is downhill — the collapse this term exists to
prevent) and from above by the hydrogen bond, which loses roughly 0.2 eV for every
0.1 Å of extra reach. What the taper buys, on a 64-water box at 997 kg/m³:

| | untapered | tapered |
| --- | --- | --- |
| intermolecular ZBL | 172 eV | **0.16 eV** |
| its contribution to $P$ | +251 kbar | **+1 kbar** |
| water dimer minimum | none, +0.60 eV at 2.91 Å | **−0.116 eV at 2.85 Å** ‡ |
| H₂ + O₂ wall, 4.0 → 0.6 Å | 21.8 eV | 20.8 eV |

‡ Every dimer figure in this section was measured **before** the intramolecular
exclusion existed, and is kept because it is what the taper radius was chosen
against. The current well is 0.1024 eV at 2.91 Å; §2.3.2 has the −0.0691 eV of
intramolecular polarization that accounts for the difference, and why recovering it
is an `eta` retune rather than a taper one.

and what it costs at the separations the bonded fit leans on, as the retained
fraction $f(r)$: H–H at 0.741 Å **0.998**, O–H at 0.958 Å **0.989**, O–O at 1.208 Å
**0.918**, and water's 1-3 H···H at 1.51 Å **0.479**. The wall is essentially
untouched where the Morse depths absorb it, which is why the change was a refit
rather than a rebuild.

$r_c$ was chosen against the dimer directly: the minimum runs −0.148 eV at 2.70 Å
($r_c = 1.4$), −0.116 eV at 2.85 Å ($1.5$), −0.090 eV at 3.00 Å ($1.6$). $w$ is
bounded by the derivative the switch itself injects, $\tfrac{1}{4w}$ times the
potential at $r_c$: at $w = 0.12$ that peak sits at 1.5 Å where O–H ZBL is 1.15 eV,
adding 2.4 eV/Å against the 17.8 eV/Å the untapered term already carries at the
O–H bond length.

The cut is in $r$, not in the reduced coordinate $x = r/a$. A fixed $x$ would be
element-transferable for free and is wrong: bond lengths scale with covalent radii
while $a$ scales as $Z^{-0.23}$, so one $x$ lands at 1.97 Å for H–H and 1.22 Å for
O–O — leaving H–H walled well past the H₂ bond while cutting O–O straight through
the O₂ bond.

$f$ is a function of $r$ alone, multiplying a pair potential that already took no
topology, so every structural guarantee below survives it unchanged.

**It is still a refit.** `fit/dissociation.py` solves against
$E_{\text{QForce}} + E_{\text{nonbonded}}$ with this term inside $E_{\text{nonbonded}}$,
so changing $r_c$ or $w$ invalidates every `.jsonl` in every dataset and
`scripts/fit.py` has to be re-run. It is not free: on HCombustion the taper lowers
the diabats at transition-state geometries — which is where close contacts are, and
therefore where the removed tail was largest — and the channel count went from 14 of
19 to **12 of 19**, losing `rxn_06`, `rxn_11` and `rxn_16`. Two of those three had
margins under 0.1 eV before it. `--frequency-weight` does not buy them back at 200×
the default, so this is structural rather than a knob setting. HCombustion is now
fitted with `--max-wavenumber 4200`, landing at 4325 cm⁻¹ (4314 once the 12-6 came
back) against the 4400 default; Water is unaffected at 3 of 3.

**Why it is topology-independent, and why that is the whole design.** The obvious
repulsion to reach for is the Lennard-Jones already in the templates, excluded
between bonded atoms the way any fixed-topology force field excludes it. That was
tried four times, and every time it failed, because q-force's 12-6 is enormous at
the separations reactive chemistry actually visits:

| pair | $r$ | what it is | 12-6 | ZBL |
| --- | --- | --- | --- | --- |
| H–H | 0.777 Å | the H₂ bond length | **504 eV** | 2.0 eV |
| O–H | 0.960 Å | the O–H bond length | **930 eV** | 5.4 eV |
| O–O | 1.210 Å | the O₂ bond length | **1348 eV** | 11.6 eV |
| O–H | 0.600 Å | the observed fusion contact | 2.6×10⁵ eV | **20.9 eV** |

A diabatic state that has *broken* a bond calls its two atoms different molecules
while they sit at the bond length, and pays hundreds of eV for it. Every attempt to
strip that penalty made the repulsion differ between diabatic states, and each one
failed differently: stripping it from the unreacted parent as well let the box fuse
at 0.60 Å; gating it on the coupling left a 19 eV plateau across rxn_16's reaction
path where a wall was half-removed; and transition states with close non-bonded
contacts came back with EVB amplitudes of −72 to −108 eV.

All four descend from one root — a pair at a bond length being charged hundreds of
eV, and the charge depending on the topology — and §2.2.1 is that root being removed
rather than its symptoms. The 12-6 is back in the force field; what follows is why
this module carries the short range and that one does not.

`ZBL` takes no topology at all. Its only parameter is $Z$, read from `Atoms.numbers`
rather than through a term list — it is the one force field here whose `__call__`
takes `numbers` instead of a `term_dict` — so there is no template, no remapping
and no per-state path by which it could acquire a state dependence. A term identical
across every diabatic state adds the same constant to every EVB diagonal;
`np.linalg.eigh` shifts the eigenvalue by exactly that and leaves the eigenvectors
untouched. So it **cannot** produce a plateau, a spurious amplitude, a pivot
dependence, or a discontinuity at $r_{\text{bimol}}$ — not "does not", cannot.

The same property means it cancels exactly out of every energy *difference*,
including the diabatic margins `fit/dissociation.py` scores channels on. It buys
stability and nothing at all for fittability; that work belongs to the bonded fit.

It is applied to bonded pairs too, since it knows nothing about bonds. Those values
(+2.3 eV on H₂, +11.6 on O₂, +11.1 on H₂O) are absorbed by the fitted Morse depths —
`fit/dissociation.py` solves against $E_{\text{QForce}} + E_{\text{nonbonded}}$ — so
every template still reproduces its own reference atomization energy to $10^{-13}$.

#### 2.2.1 The switched 12-6 (`LennardJones`, [forcefield/lj.py](forcefield/lj.py))

The taper above left the model with **no intermolecular wall at all** above 1.5 Å,
and it never had any dispersion. Measured, that put a 64-water box at −7.6 kbar at
ambient density with its equation-of-state crossing near 1250 kg/m³ against an
experimental 997 — a quarter too dense, because nothing stopped it contracting.
This term is what stops it, and it is the only non-electrostatic attraction in the
force field.

$$
u(r) \;=\; g(r)\,\cdot\,4\varepsilon\!\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right],
\qquad
g(r) \;=\; 1 - \frac{1}{1 + e^{(r - r_s)/w}},
$$

with $\sigma$ and $\varepsilon$ combined **geometrically** in both parameters (q-force's
`A=sqrt(A1*A2); B=sqrt(B1*B2)`, *not* Lorentz–Berthelot), $r_s =$ `switch_radius`
$= 0.22$ nm and $w =$ `switch_width` $=$ `taper_width / 10` $= 0.012$ nm — the
same physical width, in this module's units ([params.py](forcefield/params.py)).

$g$ is the same Fermi form as ZBL's $f$ run in the opposite direction, but it is
**not $1 - f$**: the two are centred at different radii, so
$g(r) + f(10r) \ne 1$. Calling them complementary is the mistake the next paragraph
is about. Like ZBL's, the exponent is clipped at $\pm 500$
([lj.py:178](forcefield/lj.py#L178)).

**A linear core below $0.4\sigma$** ([params.py](forcefield/params.py),
[lj.py:222](forcefield/lj.py#L222)). Below `core_fraction` $\times\ \sigma$ the
potential is continued by its own tangent,

$$
u(r) = u(r_\text{core}) + u'(r_\text{core})\,(r - r_\text{core}),
\qquad r_\text{core} = 0.4\,\sigma ,
$$

which is 0.78 Å for H–H and 1.18 Å for O–O, where the bare wall is already 451 and
1750 eV. This is not cosmetic: without it the $\sum_\text{all} - \sum_\text{near}$
decomposition below evaluates $r^{-12}$ on both sides at compressed geometries and
loses the difference to floating point — H₂ at 0.3 Å puts $10^8$ eV in each sum, at
0.2 Å $10^{10}$, at 0.024 Å $10^{21}$, and a 3000 K probe reached $10^{21}$ by step
143 with the physical difference coming back quantized to $2^{22}$ eV and the box
at $10^{16}$ K. With the continuation the same geometries read ~4500 eV and
~5700 eV. It also lets $g\,u$ underflow cleanly instead of forming $0 \times 10^{21}$.

**The switch is what makes the term usable, and its radius is the whole argument.**
Bare, this is the 500–1400 eV column in the table above. Switched at 2.2 Å:

| | bare | switched | what it is |
| --- | --- | --- | --- |
| O–H, 0.958 Å | 953 eV | **0.031 eV** | the O–H bond |
| H–H, 0.741 Å | 892 eV | **0.005 eV** | the H₂ bond |
| O–O, 1.208 Å | 1375 eV | **0.353 eV** | the O₂ bond, the worst case |
| H⋯H, 1.51 Å | 0.138 eV | 0.0004 eV | water's 1-3 pair |
| O⋯H, 1.94 Å | 0.146 eV | 0.015 eV | the hydrogen-bond contact |
| O⋯O, 2.4 Å | 0.262 eV | 0.220 eV | **the wall**, 84% retained |
| O⋯O, 2.6 Å | 0.076 eV | 0.073 eV | **the wall**, 97% retained |

Three to four orders of magnitude at a bond length, and essentially untouched where
the term does its work. That is what lets the **whole-system sum** carry no
exclusions at all: it is therefore the same number on every diabatic state, and is
added once outside the Hamiltonian exactly as `ZBL` and `ACKS2` are — where it
inherits the same guarantee, that a common shift of the diagonal moves `eigh`'s
eigenvalue by that constant and leaves the eigenvectors alone. That sum **cannot**
produce a plateau, a spurious amplitude, a pivot dependence, or a discontinuity at
$r_{\text{bimol}}$.

The intramolecular *correction* is a separate, per-state additive term and does sit
on the diagonal (§2.3.2). The distinction is the whole point: what failed four times
historically was stripping pairs out of the sum itself, which made the sum differ
between states while it was still worth hundreds of eV.

**$r_s \ne r_c$, and the gap between them is deliberate.** Setting $r_s = r_c$ is the
tidy choice — a seamless hand-over, no gap, no overlap — and it does not survive
q-force's parameters: a Fermi switch decays one $e$-fold per $w$ while $r^{-12}$ grows,
and at 1.5 Å the switch is still $1.1\times10^{-2}$ where the potential is 953 eV, so a
single water molecule collects **+20.6 eV**. What sets $r_s$ instead is the O₂ bond
from below and the O⋯O contact from above, and with $w = 0.12$ Å the window those two
leave is only about [2.1, 2.4] Å wide. The consequence is a gap from ~1.6 to ~2.0 Å in
which neither term acts: that is where the hydrogen bond lives, it is where *both*
forms are wrong — ZBL is a keV-stopping potential and q-force gives hydrogen
$\sigma_H = 1.96$ Å where every water model gives it zero — and `ACKS2` carries it
alone. That surface bound the dimer at −0.112 eV when this was written and binds it
at −0.1024 eV now, the difference being the exclusion of §2.3.2 rather than anything
in this gap. `tests/test_collapse.py` is what checks nothing squeezes through it.

**It is a refit, for the same reason the taper was.** `fit/dissociation.py` solves
against $E_{\text{QForce}} + E_{\text{nonbonded}}$ and this term is inside
$E_{\text{nonbonded}}$, so changing $r_s$, $w$ or `core_fraction` invalidates every
`.jsonl` in the dataset that states them. What that cost is recorded in
[production/density-300K/README.md](../../production/density-300K/README.md).

**The decomposition, and it is live.** LJ depends on the topology only through
exclusions, and exclusions are local to a molecule:

$$
E_{\text{LJ}}(\text{state}) \;=\; \underbrace{\sum_{i<j} u_{ij}}_{\text{topology-free}} \;-\; \underbrace{\sum_{\substack{i<j \\ d_G(i,j)\,\le\,3}} u_{ij}}_{\text{per-molecule exclusion terms}}
$$

with $d_G$ the bond-graph distance and depth 3 (`exclusion_depth`, GROMACS
`nrexcl = 3`). The calculator evaluates the first sum once on the whole system and
the second per diabatic state, through `QForce.compute_exclusion`
([qforce.py:334](forcefield/qforce.py#L334)); `forcefield/exclusions.py` derives the
terms at load and §2.3.2 has the argument. `tests/test_reference_energies.py` asserts
the two halves are the same function of the same numbers, switch and core
continuation included. (Scoring a raw `.jsonl` *without* going through
`with_exclusions` reads H₂ at +842 eV against a reference of −4.67, which is what
that helper exists to prevent.)

Two rules keep that identity real. `compute_exclusion` imports `lj.pair_potential`
rather than open-coding the form ([qforce.py:361](forcefield/qforce.py#L361)) —
open-coded, the two halves once disagreed by **1609 eV** on an H₂ template the
moment `pair_potential` gained its linear core. And `exclusion_terms` stores the
*already-combined* geometric $\sigma$ and $\varepsilon$ for the pair, skipping any
pair where either is zero ([exclusions.py:166](forcefield/exclusions.py#L166)), so
the excluded and the counted branch use identical numbers.

`compute_zblexclusion` ([qforce.py:382](forcefield/qforce.py#L382)) is the same
construction for `ZBL`, and it carries the **opposite** unit trap:
`lj.pair_potential` is unit-agnostic and can be handed q-force's nm directly, while
`zbl.pair_potential` is not — `SCREENING_LENGTH` is in Å and `zbl_ccoul` in eV·Å, so $r$
converts going in and the result converts coming back. Getting that wrong leaves the
energy right and the forces wrong by a factor of ten.
`tests/test_gradients.py::TestZBLGradients` and `tests/test_stress.py` now assert
energy, forces **and** virial cancel between the two forms to $10^{-9}$.

Named for the record: `reactive_exclusion_weights`, `fixed_basis_weights`,
`exclusion_correction` and `lost_exclusion_correction` were the four attempts at
making the *sum* state-dependent, and they are gone.

### 2.3 Electrostatics (`ACKS2`, [forcefield/acks2.py](forcefield/acks2.py))

Charge equilibration with a Kohn–Sham response block. Solve, in term order,
$A\mathbf{z} = \mathbf{b}$ with $\mathbf{z} = [\,\mathbf{Q},\ \mathbf{u},\ \lambda_{\text{tot}},\ \lambda_{\text{KS}}\,]$:

$$
A = \begin{pmatrix}
K & -I & -\mathbf{1} & 0 \\
-I & X & 0 & -\mathbf{1} \\
-\mathbf{1}^{\mathsf T} & 0 & 0 & 0 \\
0 & -\mathbf{1}^{\mathsf T} & 0 & 0
\end{pmatrix},
\qquad
\mathbf{b} = \begin{pmatrix} -\boldsymbol{\mu} \\ \mathbf{0} \\ 0 \\ 0\end{pmatrix},
$$

$$
A_{ii} \mathrel{+}= 2\eta_i,
\qquad
X_{ij} = \chi_i\chi_j\,e^{-r_{ij}/\tau_{ij}}\ (i\ne j),
\qquad \tau_{ij} = \tfrac12(\tau_i + \tau_j),
\qquad X_{ii} = -\sum_{j\ne i} X_{ij},
$$

with $K$ the charge kernel of §2.3.1 — $\operatorname{erf}(\gamma r)/r$ under open
boundaries, its lattice sum under periodic ones.

**The hardness is added to the Coulomb diagonal, not written over it**
([acks2.py:117](forcefield/acks2.py#L117)). Under open boundaries $K_{ii} = 0$ and
the two are the same operation; the periodic kernel carries an atom's interaction
with its own images there, and it belongs in the equilibration alongside $2\eta_i$.

The two $\mathbf{1}$ rows/columns enforce total-charge and KS-potential
constraints; both right-hand sides are hard-coded to zero
([acks2.py:124](forcefield/acks2.py#L124), [acks2.py:129](forcefield/acks2.py#L129)),
so the system is always **exactly neutral** — there is no net-charge parameter, and
a formally charged fragment is scored as its neutral isomer. That is a real
limitation and it has a consequence recorded in §2.3.2: it is why H₃O⁺ cannot be
fitted without the Coulomb exclusion. $A$ is symmetric, which the force adjoint
exploits.

**Parameter names.** The term kwargs are `mu` ($\mu$), `eta` ($\eta$), `soft_amp`
($\chi$) and `soft_decay` ($\tau$); there are no `chi` or `tau` keys. The kernel
width $\gamma$ is `params.gamma` $= 2.0$ Å⁻¹, named once and shared by both kernel
implementations — it was an unnamed literal `2` repeated at five sites in this file
before `ewald.py` existed.

The energy added to the system is the damped Coulomb sum only:

$$
E_{\text{elec}} = \frac{C}{2}\sum_{i,j} S_{ij}\,Q_i Q_j K_{ij},
\qquad C = 14.4\ \text{eV\,Å},
$$

with $S$ the exclusion screen of §2.3.2 ($S \equiv 1$ recovers the unscreened sum)
and the $i = j$ term retained, since $K_{ii}$ is a real self-image interaction under
periodicity. Everything `compute` returns — energy, forces and virial — is derived
from this one functional, so the three are consistent with each other whatever $S$
is, including the fractional screen an EVB ground state produces.

**Charge response.** The charges solve a geometry-dependent system with a
geometry-free $\mathbf{b}$, and $E_{\text{elec}}$ is *not* stationary in
$\mathbf{Q}$ — the full ACKS2 functional is, this Coulomb piece alone is not — so
$d\mathbf{Q}/d\mathbf{x}$ contributes. Differentiating the solve,
$\partial\mathbf{z}/\partial\mathbf{x} = -A^{-1}(\partial A/\partial\mathbf{x})\mathbf{z}$, gives

$$
\frac{dE}{d\mathbf{x}} \;=\; \left.\frac{\partial E}{\partial \mathbf{x}}\right|_{\mathbf{Q}}
\;-\; \boldsymbol{\lambda}^{\mathsf T}\frac{\partial A}{\partial \mathbf{x}}\mathbf{z},
\qquad
\boldsymbol{\lambda} = A^{-1}\frac{\partial E}{\partial \mathbf{z}},
$$

one extra solve rather than one per coordinate (`compute_response_forces`,
[acks2.py:192](forcefield/acks2.py#L192)). Only $K$ and $X$ depend on geometry;
anything geometry-dependent added to `build_system` must be differentiated here
too, or the forces stop being the gradient of the energy. Omitting this term was
the source of NVE drift for every species with nonzero charges. The response
carries a strain derivative as well as a position one and needs no separate solve:
both are the same kernel contraction under the same weight, which is what
`kernel.contract` returns as a pair.

**The screen enters this expression once, not twice** — see §2.3.2, which is the
subtlest thing in the module.

**Charges are re-solved, not frozen** — but they are cached. `prepare` keys on the
positions, the atom indices **and the cell**
([acks2.py:422](forcefield/acks2.py#L422)); the cell is in the key because a
barostat step is a different problem at identical fractional coordinates. The bond
graph is deliberately *absent* from the key: the solve does not depend on it, and a
key that pretended otherwise would throw the charges away after every reaction for
nothing. The key still does not cover the `mu` / `eta` / `soft_amp` / `soft_decay`
values, so the same geometry evaluated with different parameters would reuse stale
charges; nothing in the current pipeline does that, and `fit/dissociation.py` keys
its own nonbonded cache separately.

#### 2.3.1 The charge kernel, open and periodic ([forcefield/ewald.py](forcefield/ewald.py))

**Electrostatics is the one nonbonded term that is not short ranged**, so under full
periodicity it is summed over images rather than truncated at the nearest one. Both
forms live behind one interface and `acks2.py` never branches on `pbc`:

| call | returns |
| --- | --- |
| `kernel.matrix()` | $K$, the $(n,n)$ kernel matrix |
| `kernel.contract(W)` | $\partial S/\partial r_i$ and $\partial S/\partial\epsilon_{ab}$ for $S = \sum_{ij} W_{ij}K_{ij}$ |

`contract` is the reason the kernel is an object rather than a matrix. Both places
ACKS2 differentiates the kernel — the explicit Coulomb force and the
$-\boldsymbol\lambda^{\mathsf T}(\partial A/\partial\mathbf{x})\mathbf{z}$ response —
are of exactly that form and differ only in the symmetric weight $W$, so the
periodic derivatives are written once instead of once per caller; and the
reciprocal-space part of each is not a sum over pair separations at all.

`MinimumImage` is the open-boundary kernel, $\operatorname{erf}(\gamma r)/r$ at the
nearest image, and also the fallback for a **slab or a wire** — Ewald here needs all
three directions periodic, and a 3D lattice sum under 2D periodicity would be the
wrong sum, not an approximate one. `Ewald` is the lattice sum:

$$
K_{ij} \;=\; \sideset{}{'}\sum_{\mathbf{n}}
\frac{\operatorname{erf}\!\big(\gamma|\mathbf{r}_{ij}+\mathbf{n}|\big) - \operatorname{erf}\!\big(\kappa|\mathbf{r}_{ij}+\mathbf{n}|\big)}{|\mathbf{r}_{ij}+\mathbf{n}|}
\;+\; \frac{4\pi}{V}\sum_{\mathbf{k}\ne 0} \frac{e^{-k^2/4\kappa^2}}{k^2}\cos\big(\mathbf{k}\cdot\mathbf{r}_{ij}\big)
\;-\; \delta_{ij}\frac{2\kappa}{\sqrt{\pi}}
\;\underbrace{-\;\frac{\pi}{\kappa^2 V}}_{\text{background}} .
$$

Writing $\operatorname{erf} = 1 - \operatorname{erfc}$ twice turns the periodic sum of
the smeared kernel into a standard Ewald sum of $1/r$ minus a short-ranged
remainder, and the two real-space pieces merge into the single difference above. The
prime excludes $\mathbf{n} = 0$ when $i = j$: an atom does not interact with itself,
but it does interact with its own images.

Three things about the periodic kernel have no open-boundary analogue.

**$K_{ii} \ne 0$.** That self-image term is 80% of the rock-salt energy in
`tests/test_ewald.py`, not a rounding correction, and it is why `build_system` adds
the hardness to the Coulomb diagonal instead of overwriting it.

**The $k = 0$ term is dropped and its background put back.** The $k = 0$ term of the
reciprocal sum is the divergent one; regularizing it against a neutralizing
background leaves the constant $-\pi/(\kappa^2 V)$ in *every* entry of $K$, diagonal
included. Dropping it is legal only because `build_system` constrains
$\sum_i q_i = 0$: a constant on every entry of $K$ contributes that constant times
$\big(\sum_i q_i\big)^2$ to the energy and that constant times $\sum_i q_i$ to each
ACKS2 row. That made every **neutral** contraction right and every individual
$K_{ij}$ meaningless — an $K_{ij}$ drifted with the splitting parameter by 0.0054 on
an O–H pair between `accuracy` $10^{-4}$ and $10^{-12}$, while the neutral
contraction held to ten digits.

The intramolecular exclusion (§2.3.2) is the first thing in this codebase to
contract $K$ against a **non-neutral** weight — the excluded pairs of a water carry
$\sum_{\text{excl}} q_iq_j \ne 0$ — so `Ewald.background` is now carried explicitly.
It changes no energy, force, virial or charge that predates it, by the same
$\big(\sum_i q_i\big)^2$ argument, and every neutral contraction is bit-identical
across the change. It surfaced first in the **virial**, at 0.5% of a seven-atom box,
because $\kappa$ is a function of the cell and a strain moves it; the energy was
wrong by the same mechanism without moving. In `contract` the background reaches the
strain through the volume alone — $V \to V(1 + \operatorname{tr}\boldsymbol\epsilon)$
puts $-\,\texttt{background}\cdot\sum_{ij}W_{ij}$ on the diagonal — and has no
position derivative at all. $\kappa$ is deliberately *not* differentiated: the
completed kernel is independent of it to the reciprocal sum's own accuracy, which is
the property the background restores.

`tests/test_ewald.py::TestConvergence::test_independent_of_the_splitting_for_a_charged_weight`
pins this, and carries its own anti-vacuity guard asserting that the *un*backgrounded
kernel does drift (measured spread $9.09\times10^{-4}$ against a guard at
$5\times10^{-4}$). One sentinel moved with the fix and is not a regression:
`test_self_image_term_is_what_makes_it_work` went 0.3502 → 0.3289, because zeroing
the diagonal is itself a non-neutral weight and the background no longer cancels
against it.

**Reciprocal vectors strain inversely to positions**, $\mathbf{k} \to (I - \boldsymbol\epsilon)\mathbf{k}$,
so $\partial k^2/\partial\epsilon_{ab} = -2k_ak_b$ — that is where the reciprocal
virial comes from. The structure factor is invariant under the same strain, since
$\mathbf{k}\cdot\mathbf{r}$ is, so there is no $v_av_b$ term to match the real-space
one.

**`accuracy` $= 10^{-8}$** sets $\kappa$ and the reciprocal cutoff together, so both
halves converge to the same level. Loosening it is cheap in the reciprocal vector
count — which scales as $(-\log a)^3$ — and expensive in the **stress**, for the
reason its comment gives: $\kappa$ is derived from the cell, so a strained cell is
summed with a slightly different splitting, and the residual
$\partial(\text{truncation error})/\partial(\text{cell})$ has no counterpart in the
analytic expression. That error is amplified by $2\log(1/a)$ relative to the
truncation error itself. Reciprocal vectors are selected on an **integer ellipsoid**
rather than by $|k|$, so the set is piecewise constant in the cell and a strain
cannot move a whole degenerate shell across the cutoff. Cost is +24 ms on a 200-atom
box against a 200–450 ms force call.

**Only the Coulomb block is summed over images.** ACKS2's other geometry-dependent
block, the bond softness, decays as $e^{-r/\tau}$ with $\tau \approx 0.3$ Å; at the
nearest-image cutoff of a 9 Å cell that is $e^{-15}$, so it stays a nearest-image
pair term at every boundary condition and `contract_pairs` is shared with it.

**None of this invalidated any `.jsonl`**, unlike the three radii of §2.2: every
dataset template carries `pbc="F F F"`, so `fit/dissociation.py` sees the unchanged
open-boundary kernel. The bit-identical `energy_bonded` in `test_performance.py`'s
reference block is the evidence.

#### 2.3.2 The intramolecular exclusion ([forcefield/exclusions.py](forcefield/exclusions.py))

**What it is for.** Without exclusions a template's gas-phase geometry is set by the
bonded terms *and* by whatever the three whole-system nonbonded sums happen to
contribute at bond lengths, and `fit/dissociation.py` has to absorb the difference
into the Morse depths. For H₃O⁺ that absorption failed outright: the fitted depth
came out at 5.76 eV against a 5.21 eV tapered-ZBL step per O–H, so the isolated
cation's true minimum sat at **1.60 Å** and the symmetric $C_{3v}$ structure was a
*saddle*. With the intramolecular nonbonded removed, a small template's gas-phase
minimum is the q-force potential's own minimum and nothing has to cancel anything.

The scale, on one water with the dataset's own parameters: the entire unscreened
ACKS2 energy of an isolated H₂O is **−4.64419 eV**, every bit of it intramolecular,
and the screen takes it to exactly 0.

`exclusions.exclusion_terms` derives one term per pair within `exclusion_depth = 3`
bonds (GROMACS `nrexcl = 3`), at load time
([core/reactionset.py:183](core/reactionset.py#L183)), for all three sums:

| term | evaluated by | note |
| --- | --- | --- |
| `exclusion` | `QForce.compute_exclusion` | 12-6; emitted only where both $\sigma$ and $\varepsilon$ are nonzero, since $\sigma_{\mathrm H} = 0$ makes the pair identically zero and a term would be an expensive no-op |
| `zblexclusion` | `QForce.compute_zblexclusion` | needs atomic numbers, which ZBL reads from the `Atoms` rather than a template, so they are handed in |
| `coulombexclusion` | `ACKS2`, **not** `QForce` | carries no kwargs at all — the charges are not known at load time |

**Why the exclusions sit on the EVB diagonal when the sums do not.** Which pairs are
1-2/1-3/1-4 is exactly what a diabatic state disagrees about. So the whole-system
sums stay where they were — one evaluation, outside the Hamiltonian, the same number
on every state, §2.2 and §2.2.1 unchanged — and the *correction* is what becomes
per-state:

$$
E(\text{state}) \;=\; \underbrace{\sum_{i<j} u_{ij}}_{\text{topology-free, once}} \;-\; \underbrace{\sum_{\substack{i<j \\ d_G(i,j)\,\le\,3}} u_{ij}}_{\text{per-state term}}
$$

with $d_G$ the bond-graph distance. For ZBL and the 12-6 this is an ordinary
additive pair correction and nothing about the architecture changes: `QForce`
evaluates it alongside the bonded terms, and it enters $H_{ii}$ through the same
path a bond does.

**The 12-6's correction is small now, and that is exactly why this is safe where the
four historical attempts were not.** A diabatic state that has broken a bond pays
its `exclusion` at the bond length: 0.005–0.35 eV in §2.2.1's switched column, not
the 500–1400 eV of the bare form. A diagonal that moves by a third of an eV between
states is a chemical decision; one that moves by 900 eV is the failure mode
`lj.py`'s docstring enumerates. The switch is what made the same decomposition
usable again.

**Coulomb is not additive, and that is the whole difficulty.** ACKS2's charges come
from a linear solve whose matrix *contains* the kernel, so the exclusion cannot
simply be subtracted from the energy and left out of the equilibration. The obvious
repair is to mask the kernel inside the solve. It is perfectly consistent — a
screened kernel is just another kernel in §2.3.1's sense, with energy, forces and
virial all derived from it — and it is **wrong here for a reason that has nothing to
do with consistency**: it makes the charges a function of the bond graph.
`System.calculate` solves them once, outside the Hamiltonian, *precisely because*
they are the same problem for every diabatic state; topology-dependent charges
solved once anyway make the total energy depend on which state happened to seed the
block, measured at **0.88 eV** on `tests/test_evb_invariants.py`. Solving them per
state instead would be correct, and would multiply the most expensive thing in a
force call by the basis size.

What is done instead is one quantity applied in two places.

**(a) A scalar on each state's diagonal** (`ACKS2.exclusion_energy`,
[acks2.py:310](forcefield/acks2.py#L310)):

$$
C(T_s) \;=\; -\,C_{\text{coul}}\!\!\!\sum_{(i,j)\,\in\,\text{excl}(T_s)}\!\!\! Q_i\,Q_j\,K_{ij} ,
$$

at the charges `prepare` has already solved — one lookup per excluded pair into a
kernel matrix already built and memoized. That is the whole cost, and it is what
lets the exclusion decide which bonding pattern is lower, which is the point of
putting anything on a diagonal.

**(b) One screen on the whole-system energy**, after the diagonalization
(`ACKS2.screen_matrix`, [acks2.py:330](forcefield/acks2.py#L330)):

$$
S \;=\; 1 \;-\; \sum_{s} w_s\,M_s ,
$$

$M_s$ the symmetric 0/1 mask of state $s$'s excluded pairs and $w_s = c_s^2$ its
ground-state weight. A pair bonded in every state of its block is removed outright;
one bonded in only some of them is removed in proportion.

**Fractional entries are the normal case, and they are not an interpolation.**
$\sum_s w_s\,\partial C(T_s)/\partial\mathbf{x}$ is linear in the masks and
`kernel.contract` is linear in $W$, so the entire Hellmann–Feynman sum collapses into
a single weight matrix. **One contraction and one adjoint solve serve the whole
system**, which is the same cost the unexcluded term always had. Measured on a
64-water box: 65.9 ms per force call with the Coulomb exclusion on against 67.5 ms
with it off — within noise — while the exclusion itself is worth 240 eV on that box.
$S$ is held fixed under the derivative, which is correct and not an omission: the
weights are eigenvector components, and a Hellmann–Feynman gradient does not
differentiate those.

**Blocks are atom-disjoint**, which is what makes the subtraction safe.
`ReactionNetwork.reaction_blocks` yields one entry per reaction group plus one
$(\varnothing, \{\text{molecule}\})$ entry per untouched molecule, so every atom is
in exactly one block and a block's weights sum to 1. A pair bonded in every state
therefore comes out at exactly 0, and the arithmetic never runs past it into
negative territory.

**The ordering.** `ACKS2.prepare` solves the unmasked system *before* the blocks are
diagonalized; `ACKS2.compute` evaluates the screened functional *after*. In between,
`System.calculate` adds $C(T_s)$ to each diagonal, diagonalizes, and then
**subtracts $\sum_s w_s C(T_s)$ back out of the bonded accumulation**
([system.py:186](system.py#L186)): $\varepsilon_0$ carries the correction and so does
the screened electrostatic term, and it must be counted once. The forces need no
such subtraction — `fham` never had the corrections in it, by design, because their
gradient is precisely what the single screened `compute` call supplies.

**The one asymmetry in the response, and it is the subtlest thing in the module.**
Inside `compute_response_forces` the screen goes on $\partial E/\partial\mathbf{z}$
and **not** on the $-\boldsymbol\lambda^{\mathsf T}(\partial A/\partial\mathbf{x})\mathbf{z}$
weight. The energy is the screened one, so its derivative with respect to the charges
carries the screen; $A$ was built from the *unscreened* kernel, so $\partial A/\partial\mathbf{x}$
is the unscreened kernel derivative and the weight it is contracted against must not
be masked. Screening both would be the gradient of a functional whose charges were
also screened, which is not the functional being evaluated; screening neither drops
the exclusion from the response entirely. **Both mistakes produce plausible forces
and an NVE trajectory that drifts**, and nothing else
([acks2.py:219](forcefield/acks2.py#L219)).

**Verification.**

| | |
| --- | --- |
| pivot invariance, seeded from every state of a converged basis | $\Delta E = -7.1\times10^{-15}$ eV, $\Delta F = 5.3\times10^{-15}$ eV/Å |
| `System` forces vs central differences, block genuinely mixed at weights 0.679 / 0.321 | $8.95\times10^{-8}$ eV/Å against $\|F\|_{\max} = 17.9$ |
| screened `ACKS2` alone — forces / virial / virial asymmetry | $1.0\times10^{-9}$ / $1.4\times10^{-7}$ / $4.4\times10^{-16}$ |
| NVE spread at $dt = 0.20 / 0.10 / 0.05$ fs, screen live at weights $\approx 0.50/0.50$ | 39.394 / 10.004 / 2.497 meV — ratios 3.94, 4.00, clean $O(dt^2)$ |

`tests/test_evb_invariants.py::TestPivotInvariance::test_nonbonded_is_independent_of_pivot`
is the standing guard; `tests/test_gradients.py` carries the fractional-screen force
case (with a deliberately irrational, non-0/1 symmetric screen), a check that the
screen does not move the charges at `atol = rtol = 0`, and a check that screening a
pair removes exactly `exclusion_energy` of it.

**What it costs, and it is real.** The charges are exactly what they were before
exclusions existed — an isolated water still gives $q_{\mathrm H} = +0.30399$ — so no
molecular dipole moves. What the exclusion does remove is the *intramolecular half of
the polarization response*, and that is worth 0.07 eV of the hydrogen bond.
Decomposed on the dimer at $R_{\text{OO}} = 2.85$ Å with the datasets held fixed, so
this is the mechanism and not a refit:

| | |
| --- | --- |
| dimerization gain, unscreened ACKS2 | −0.2063 eV |
| of which intramolecular self-energy | **−0.0691 eV** ← removed here |
| $q_{\mathrm H}$, monomer → dimer | +0.30399 → +0.31264 |

The molecules polarize each other, the charges grow, and the intramolecular Coulomb
energy falls along with the intermolecular one. Booking that gain inside the molecule
is exactly what the exclusion exists to stop — the bonded terms own the intramolecular
energy, and a template whose gas-phase minimum depends on its neighbours is the thing
this module was written to remove — but the binding goes with it: the dimer well runs
**0.1677 eV at 2.85 Å → 0.1024 eV at 2.91 Å**. `eta` is the lever to recover it, since
the intermolecular term scales as $q^2$ and softening the hardness buys depth back
without reintroducing the coupling. **That retune is outstanding**, and until it is
done `tests/test_water_structure.py`'s dimer assertions fail by construction.

**`exclude_coulomb`** ([params.py](forcefield/params.py)) turns the
Coulomb exclusion off and keeps the other two. The result is a dataset the whole
pipeline still evaluates consistently and **cannot fit**: the solve is hard-constrained
to zero total charge, so ACKS2 scores H₃O⁺ as a neutral H₃O, and the −4.8 eV of
spurious binding that produces has nothing left to cancel it once ZBL's intramolecular
+15.45 eV is excluded.

### 2.4 The virial, and the stress the calculator publishes

Every force field here returns a third quantity alongside energy and forces: the
virial

$$
W_{ab} \;=\; \frac{\partial E}{\partial \epsilon_{ab}},
$$

the derivative with respect to a homogeneous strain applied to the cell with the
atoms carried along affinely. It needs no new derivatives. Every energy in the
model is a function of minimum-image displacement vectors alone, so a strain maps
$\mathbf{v} \to (I + \boldsymbol{\epsilon})\mathbf{v}$ and

$$
W_{ab} \;=\; \sum_{\text{vectors}} v_a \left(\frac{\partial E}{\partial \mathbf{v}}\right)_b ,
$$

which is one outer product per displacement vector over the same gradient the
forces are already scattered from (`QForce._virial`,
[qforce.py:116](forcefield/qforce.py#L116)). For a pair term this reduces to

$$
W_{ab} \;=\; \tfrac12 \sum_{i \ne j} \frac{1}{r_{ij}}\frac{du}{dr}\, v_a v_b ,
$$

the form `ZBL`, `LennardJones` and `ACKS2.compute_coulomb` all use. **The
prefactors differ by term and are not interchangeable:** the $\tfrac12$ that
cancels for forces (because $\mathbf{v}_{ij} = -\mathbf{v}_{ji}$ flips sign) does
*not* cancel here, since $v_a v_b$ is even under $i \leftrightarrow j$. ACKS2's
charge-response block carries a factor of 1 against $\tfrac12$ on its energy and 2
on its forces, for the same reason. Both asymmetries now live in one place,
`ewald.contract_pairs` ([ewald.py:94](forcefield/ewald.py#L94)), which the softness
block and both kernels share.

`EVBCoupling` is the exception, because an RMSD to a stored template is not a
function of displacement vectors. Its virial is formed once at the top level over
absolute positions ([coupling.py:47](forcefield/coupling.py#L47)):

$$
W_{ab} \;=\; -\sum_{n} x_{n,a}\, F_{n,b} .
$$

This is origin-independent because `Superpose3D` removes the centroid, so
$\sum_n \partial E/\partial\mathbf{x}_n = 0$. The term carries a stress at all only
because the Kabsch fit includes a **scale** factor $s$ — an RMSD to a fixed
template is not scale-invariant, so uniformly inflating the cell does change it.

**Unit trap.** The virial is an energy. In `QForce` and `LennardJones` it converts
with `units.kJ / units.mol` and **no** length factor, unlike the forces, because
$\mathbf{v}$ is already in nm and $\partial E/\partial\mathbf{v}$ in kJ/mol/nm
([qforce.py:96](forcefield/qforce.py#L96),
[lj.py:414](forcefield/lj.py#L414)). An extra `/ units.nm` is wrong by exactly a
factor of 10 and surfaces only as a pressure that is silently an order of
magnitude out; `tests/test_stress.py::TestQForceStress::test_the_unit_conversion`
exists to catch it.

The virial is threaded through the EVB machinery exactly as the forces are —
`Block` stores `virials` and `coupling_virials`, `Block.hamiltonian()` returns
`(ham, fham, vham)`, and `System.calculate` contracts $\mathbf{W}$ with the
ground-state eigenvector alongside $\mathbf{F}$ (§4.1). One sign asymmetry is
deliberate and worth remembering: `fham` holds **forces** ($-\partial E/\partial
\mathbf{x}$) while `vham` holds **virials** ($+\partial E/\partial
\boldsymbol{\epsilon}$), which is why the switching chain rule of §1.4 appears with
a $-$ in the position line and a $+$ in the strain line
([basis.py:180](basis.py#L180), [basis.py:605](basis.py#L605)).

`System` returns the virial as a single 3×3 for the whole system, not a stress —
keeping it unnormalized means a non-periodic (zero-volume) system is still
well-defined. `ase.py` does the division ([ase.py:79](ase.py#L79)):

$$
\sigma_{ab} \;=\; \frac{1}{V} W_{ab},
$$

with **no sign flip** — $\sigma = V^{-1}\,\partial E/\partial\epsilon$ is already
ASE's convention — converted to Voigt-6 by
`ase.stress.full_3x3_to_voigt_6_stress`. At $V = 0$ the key is simply absent,
which is what ASE expects of an unavailable property. `stress` is in
`DynamicTopology.implemented_properties`; without it every ASE barostat raises
before taking a step.

Verification is `tests/test_stress.py`: central differences in the *cell* with
atoms scaled affinely, one case per `compute_*`, plus a symmetry assertion on
every virial (a transposed contraction passes a trace-only pressure check) and two
sign anchors — ZBL must give a positive pressure, and the 12-6 must flip sign
across its minimum.

---

## 3. Coupling force field

The off-diagonal is a Gaussian in the optimally superposed RMSD to a stored
ensemble of transition-state reference geometries (`compute_rmsd`,
[forcefield/coupling.py:50](forcefield/coupling.py#L50)):

$$
V(\mathbf{x}) \;=\; \frac{A}{M}\sum_{m=1}^{M} \exp\!\big(-a\,\rho_m^2\big),
\qquad
\rho_m = \min_{R,\mathbf{t},s}\sqrt{\frac{1}{N}\sum_{k}\big\|\mathbf{y}^{(m)}_k - (s R\,\mathbf{x}_k + \mathbf{t})\big\|^2},
$$

the minimization being the `superpose3d` Kabsch fit over rotation $R$,
translation $\mathbf{t}$ and scale $s$. Its gradient:

$$
\frac{\partial \rho_m}{\partial \mathbf{x}_k} = \frac{1}{N\rho_m}\Big(sR\,\mathbf{x}_k + \mathbf{t} - \mathbf{y}^{(m)}_k\Big)\,(sR),
\qquad
\mathbf{F} = \frac{1}{M}\sum_m 2\,a\,\rho_m\,A e^{-a\rho_m^2}\,\frac{\partial \rho_m}{\partial \mathbf{x}} .
$$

Rows are matched by *template* index: the live atoms are ordered by
$\pi(1), \pi(2), \dots$ rather than by the isomorphism matcher's discovery order,
or the superposition would compare atoms against the wrong reference positions.
Under PBC the fragment is unwrapped by cumulative minimum-image displacement
before superposition.

Unlike `QForce`, the `compute_*` methods here return 2-tuples $(E, \mathbf{F})$;
the virial is formed once in `__call__` (§2.4).

Channel couplings are memoized per geometry on $\big(h(\text{reaction}),\ \pi\big)$
in `EVBBasis._coupling` ([basis.py:468](basis.py#L468)), which also fixes the row
order to `sorted(mapping)` before the superposition.

### Fitting $A$ and $a$ ([fit/coupling.py](fit/coupling.py))

Both parameters are fixed by the three frames every `rxn_*.xyz` already carries.

**Amplitude** — at the transition state $\rho = 0$, so $V(\mathbf{x}_{\text{TS}}) = A$
exactly, and the $2\times2$ secular equation inverts in closed form against the
reference barrier $E^\star$:

$$
A \;=\; -\sqrt{\big(\bar{H} - E^\star\big)^2 - \eta^2},
\qquad
\bar{H} = \tfrac12\big(H_R + H_P\big),\quad
\eta = \tfrac12\big(H_R - H_P\big),
$$

all evaluated at $\mathbf{x}_{\text{TS}}$ on the dataset's free-atom zero. The
sign is a phase choice — only $A^2$ enters the ground root. A real solution
requires $E^\star < \min(H_R, H_P)$: the adiabatic ground state lies below every
diabat by construction, so a violation means the *diabatic* energies are wrong,
not the barrier.

**Width** — the coupling must vanish where the diabatic picture is already
correct, i.e. at the endpoint minima. Requiring $|V| \le \epsilon$ at whichever
endpoint is *closer* to the TS:

$$
a \;=\; \frac{\ln\big(|A|/\epsilon\big)}{\rho_{\min}^2},
\qquad
\rho_{\min} = \min\big(\rho(\mathbf{x}_R, \mathbf{x}_{\text{TS}}),\ \rho(\mathbf{x}_P, \mathbf{x}_{\text{TS}})\big),
$$

with $\epsilon = $ `DEFAULT_EPS` $= 10^{-3}$ eV. The reference frame is the middle
frame of the `.xyz`, and $\rho_{\min}$ the smaller of the two endpoint RMSDs
([fit/coupling.py:60](fit/coupling.py#L60)). When $A = 0$ the width is fitted
against `NOMINAL_AMPLITUDE` $= 1.0$ instead, so a decoupled channel still gets a
finite, meaningful $a$.

Each fitted term is tagged with a `provenance`
([fit/coupling.py:166](fit/coupling.py#L166)):

| `provenance` | meaning |
| --- | --- |
| `fitted` | $A$ inverted from the reference barrier |
| `decoupled` | barrier not invertible; $A = 0$, no stabilization, so `EVBBasis` never admits the state at all |
| `placeholder` | $A$ supplied by hand |

`Block.placeholder_channels` reports any channel entering a basis on a
placeholder. The shipped uniform placeholders were $A = -10$ eV, $a = 10$ Å⁻²,
which is still 0.96–7.70 eV at the endpoints of the nineteen channels — the reason
`PLACEHOLDER_AMPLITUDE` is recognized and reported rather than silently used.

---

## 4. The EVB scheme

### 4.1 Block Hamiltonian

For a block with closed basis $\{T_1,\dots,T_n\}$ (`Block.hamiltonian`,
[basis.py:176](basis.py#L176)):

$$
H_{ii} \;=\; E^{\text{QForce}}(T_i) \;+\; C(T_i),
\qquad
H_{ij} \;=\; w_{ij}\,V_{ij} \quad (i \ne j).
$$

$E^{\text{QForce}}(T_i)$ is the bonded energy **including** the `exclusion` and
`zblexclusion` corrections, which are ordinary additive terms and reach the diagonal
through the same path a bond does. $C(T_i)$ is the Coulomb exclusion of §2.3.2,
which cannot go through `QForce` because it needs the charges; `System.calculate`
adds it here and takes it back off the bonded accumulation afterwards (§4.2).

Every admitted pair carries an off-diagonal — this is a **full** matrix, not a
star. Where several symmetry-equivalent mappings reach the same product topology
they are one state reached by equivalent routes, so the most strongly coupled
representative is kept (compared on $|wV|$, since that is what enters the matrix,
and choosing on coupling rather than enumeration order keeps the choice
independent of atom labelling).

Diagonalize and take the lowest root:

$$
H\mathbf{c}^{(\alpha)} = \varepsilon_\alpha \mathbf{c}^{(\alpha)},
\qquad
\varepsilon_0 = \mathbf{c}^{\mathsf T} H \mathbf{c},
\qquad
\mathbf{F} = -\frac{\partial \varepsilon_0}{\partial \mathbf{x}} = \sum_{i,j} c_i c_j \,\mathbf{F}^{H}_{ij},
\qquad
\mathbf{W} = \frac{\partial \varepsilon_0}{\partial \boldsymbol{\epsilon}} = \sum_{i,j} c_i c_j \,\mathbf{W}^{H}_{ij},
$$

the last two by Hellmann–Feynman, with
$\mathbf{F}^H_{ij} = -\partial H_{ij}/\partial\mathbf{x}$ and
$\mathbf{W}^H_{ij} = +\partial H_{ij}/\partial\boldsymbol{\epsilon}$ (diagonal: state
forces and virials; off-diagonal: the switched coupling gradients of §1.4). The
eigenvector is stationary, so only the matrix's explicit dependence survives at
first order — the same argument covers both. Implemented as the three einsums

```python
energy_gs = np.einsum("i,ij,j->",     statevec, ham,  statevec)
forces_gs = np.einsum("i,ijnd,j->nd", statevec, fham, statevec)
virial_gs = np.einsum("i,ijab,j->ab", statevec, vham, statevec)
```

([system.py:172](system.py#L172)). A single-state block skips the eigenproblem
entirely and takes `energies[0]`, `forces[0]`, `virials[0]` directly.

The block reports the adiabatic gap $\varepsilon_1 - \varepsilon_0$, the weights
$c_i^2$, `min_switch` (the smallest $w$ on any channel, so a caller can see
whether the surface is inside a ramp), `depth`, `capped`, `basis_size` and
`placeholder_channels`; `System` collects these per block under `results["blocks"]`
([system.py:197](system.py#L197)).

### 4.2 Total energy

$$
\boxed{\;
E_{\text{total}} \;=\; \sum_{B}\Big[\varepsilon_0^{(B)} - \textstyle\sum_{s} w_s\,C(T_s)\Big]
\;+\; E_{\text{elec}}^{\text{ACKS2}}\big[S\big]
\;+\; \sum_{i<j} u^{\text{ZBL}}_{ij}
\;+\; \sum_{i<j} u^{\text{12-6}}_{ij}
\;}
$$

with $\mathbf{F}$ and $\mathbf{W}$ summed the same way, term for term. The last
three are topology-independent and evaluated once on the whole system
([system.py:227](system.py#L227)–[system.py:257](system.py#L257)), none of them on a
diagonal. That is legitimate because a constant shift of every diagonal shifts the
ground eigenvalue by exactly that constant — the couplings here come from
`EVBCoupling` and do not depend on the diagonal at all — and it requires each of the
three sums to actually *be* the same number on every state, which §2.2, §2.2.1 and
§2.3 each argue for their own term.

**The two correction terms are one quantity, counted once.** $C(T_s)$ went onto
state $s$'s diagonal before the diagonalization, so $\varepsilon_0^{(B)}$ carries
$\sum_s w_s C(T_s)$; the screen $S$ puts the same amount into
$E_{\text{elec}}^{\text{ACKS2}}$. Subtracting it back out of the block sum
([system.py:186](system.py#L186)) is what keeps `energy_bonded` the bonded energy and
stops the exclusion being double counted. The **forces carry no such subtraction**:
`fham` never held the corrections, because their gradient is exactly what the single
screened `ACKS2.compute` call supplies. Getting that asymmetry backwards is
energy-conserving and wrong, or force-consistent and double counted, depending on
which half is changed.

The ZBL and 12-6 exclusions need no counterpart to any of this: they are ordinary
additive terms inside $E^{\text{QForce}}(T_i)$, and their whole-system sums were
never screened.

`System.calculate` returns `energy`, `forces`, `virial`, `topology`, and the
decomposition `energy_bonded` / `energy_nonbonded` / `energy_zbl` / `energy_lj`
plus `blocks` ([system.py:266](system.py#L266)). `ase.py` publishes `energy`,
`forces` and `stress` as ASE properties and hangs the rest — together with
`topology_changed` — off `calc.diagnostics`; the updated topology itself lives on
`calc.system.topology` rather than in `results` ([ase.py:63](ase.py#L63)).

### 4.3 Topology propagation

The pivot — the state whose topology is carried into the next MD step — is the
dominant diabat with hysteresis against the incumbent (`System._pivot`,
[system.py:68](system.py#L68)), the incumbent being `block.seed_index`:

$$
p \;=\;
\begin{cases}
\arg\max_i c_i^2, & c^2_{\max} - c^2_{\text{incumbent}} > \texttt{PIVOT\_HYSTERESIS} \\
\text{incumbent}, & \text{otherwise}
\end{cases}
\qquad \texttt{PIVOT\_HYSTERESIS} = 0.1 .
$$

Because the basis is seed-independent, this choice does not affect the energy —
it is bookkeeping that stops the trajectory label thrashing between
near-degenerate states. (The former rule, "swap when $c^2 > 0.9$", was
unreachable under a star Hamiltonian, whose ground state
$(|0\rangle - |u\rangle)/\sqrt{2}$ pins the pivot weight at exactly $1/2$ however
strong the coupling. With every off-diagonal filled, weights are free to
concentrate.)

Block topologies are merged into the system topology and returned as
`results["topology"]`, which the calculator feeds back into `System` — this is
how bonds break and form across MD steps.

---

## 5. Parameterization

### 5.1 Reference shift (`reactionset._reference_term`, [core/reactionset.py:27](core/reactionset.py#L27))

The dataset supplies an atomization energy per template (eV, referenced to free
atoms, hence exactly 0 for a free atom). Morse bonds already account for
$-\sum D$ of it at the minimum, so the stored residual is

$$
E_0 \;=\; E_{\text{atomization}} \;+\; \sum_{\text{bonds}} D .
$$

Morse therefore carries the physics, and the shift only corrects for q-force
having fitted each bond locally rather than to the molecule's total atomization
energy — a few tenths of an eV for most HCombustion templates, $+2.59$ eV for
H$_2$O$_2$.

### 5.2 The bonded fit ([fit/dissociation.py](fit/dissociation.py))

Three layers, run in that order by `scripts/fit.py`. Layers 1 and 2 are *solves*;
only layer 3 is an optimization.

**(1) Depths — `fit_dissociation_energies` ([fit/dissociation.py:462](fit/dissociation.py#L462)).**
A template's Morse depths are scaled by a single factor $\lambda$ so that the bonds
themselves reproduce the atomization energy, with the shift zeroed:

$$
\text{solve}_\lambda:\quad E_{\text{FF}}\big(\mathbf{x}_{\text{eq}};\ \{\lambda D\}\big) \;=\; E_{\text{atomization}} ,
$$

by `brentq` over `SCALE_BRACKET` $= (0.05, 20.0)$. Note $E_{\text{FF}}$ here is
**bonded plus nonbonded** — `bonded_energy` includes ACKS2, ZBL and the switched
12-6 (memoized in `_NONBONDED_CACHE`), which is why every change to
`taper_radius`, `taper_width`, `switch_radius` or `core_fraction` forces a refit.

**The exclusions are inside that target too, and adding them was a refit of both
datasets.** The fit path reaches ACKS2 through `ACKS2.__call__`, which screens by
`screen_from_terms` — the single-topology case of §2.3.2's screen — so a template is
scored with its own intramolecular electrostatics removed, and the ZBL and 12-6
exclusions arrive as ordinary terms in the same term list. That is the point: it is
what makes a small template's fitted minimum its own rather than a cancellation
against three whole-system sums. `exclude_coulomb` and `exclusion_depth` therefore
join the list above.

**(2) Bond lengths — `fit_bond_lengths` ([fit/dissociation.py:837](fit/dissociation.py#L837)).**
One equation per bond type: displace $r_0$ so the *total* stretch force vanishes at
the reference geometry,

$$
\text{solve}_{r_0}:\quad \tfrac12\big(\mathbf{F}_i - \mathbf{F}_j\big)\cdot\hat{\mathbf{u}}_{ij} \;=\; 0 ,
$$

which exists because the nonbonded terms are no longer zero at a bond length — the
Morse now has to lean into a real repulsion rather than sit at its own minimum.
This is not a sign test on the search window: the Morse pull peaks at $D\alpha/2$
and falls off past the inflection, so there are two roots, and the bracket runs
from `argmin` outward. The window is $r_0 \pm$ `MAX_LENGTH_SHIFT` $= 0.03$ nm — a
tripwire, not a working range; real shifts are 0.005–0.014 nm.

Layers 1 and 2 are coupled (moving $r_0$ changes the energy, rescaling $D$ changes
the force), so `fit_template` ([fit/dissociation.py:975](fit/dissociation.py#L975))
alternates them `LENGTH_DEPTH_ROUNDS` $= 4$ times. The residual force falls
1.5e-3 → 1.2e-4 → 1.9e-5 → 1.9e-6 eV/Å.

**(3) Shape and stiffness — `fit_force_constants` ([fit/dissociation.py:1395](fit/dissociation.py#L1395)).**
A Powell search over three per-bond-type variables — the shape *fraction*
$u = c/c_\mathrm{max}(b)$, the log force-constant scale, and the shape decay $b$ —
with layers 1 and 2 re-solved inside every objective evaluation. `mode` selects
which of the three blocks are free (`"shape"`, `"k"`, `"both"`; default `"both"`).

$$
J \;=\; \underbrace{\sum_r \max\!\big(0,\ m - m_r\big)^2}_{\text{channel margins}}
\;+\; \underbrace{\gamma \sum_b F_b^2}_{\text{leftover force}}
\;+\; \underbrace{\kappa \sum_b \max\!\big(0,\ \tilde\nu_b - \tilde\nu_{\max}\big)^2}_{\text{timestep}}
\;+\; \underbrace{\omega \sum_b \big(x_b/\text{box}_b\big)^2}_{\text{regularizer}}
$$

Three of the four are hinges: they cost nothing until a constraint is violated.
The last pulls toward "the unchanged force field" — $u = 0$, zero log $k$-scale,
$b = $ `shape_decay`.

| symbol | option | constant | default |
| --- | --- | --- | --- |
| $m$ | `--margin` | `DEFAULT_MARGIN` | 0.02 eV |
| $\gamma$ | — | `DEFAULT_GEOMETRY_WEIGHT` | 10.0 |
| $\kappa$ | `--curvature-weight` | `DEFAULT_CURVATURE_WEIGHT` | $10^{-2}$ |
| $\tilde\nu_{\max}$ | `--max-wavenumber` | `DEFAULT_MAX_WAVENUMBER` | 4400 cm⁻¹ |
| $\omega$ | `--frequency-weight` | `DEFAULT_FREQUENCY_WEIGHT` | 0.005 |
| $k$-scale bound | `--max-k-scale` | `DEFAULT_MAX_SCALE` | 2.0 (1.41× in $\tilde\nu$) |
| $u$ bound | `--max-shape` | `DEFAULT_MAX_SHAPE_FRACTION` | 1.0 |
| $b$ bounds | `--min-decay` / `--max-decay` | `DEFAULT_MIN_DECAY` / `DEFAULT_MAX_DECAY` | 1.5 / 8.0 |

$\tilde\nu_{\max}$ is a **timestep** constraint, not a spectroscopic one:
$33356/(15\,dt)$ is 4450 cm⁻¹ at $dt = 0.5$ fs, rounded to H₂'s experimental 4401.
It is evaluated on the *total* curvature (`stretch_curvatures`), bonded plus
nonbonded, since that is what actually sets the fastest mode:
$\tilde\nu = \frac{1}{2\pi c}\sqrt{k_\text{tot}/\mu}$.

**The shape bound `shape_bound(b)` ([fit/dissociation.py:1150](fit/dissociation.py#L1150)).**
$c$ is bounded by the requirement that the curve still dissociate downhill.
Setting $dE/dr \ge 0$ on the stretched branch gives

$$
c \;\le\; \min_{s > 3/b}\ \frac{2\big(e^{(b-1)s} - e^{(b-2)s}\big)}{s^2\,(bs - 3)} ,
$$

evaluated on a 200k-point grid and memoized, not solved. The bound moves steeply
with $b$ — 0 at $b=1$, 0.180 at $1.5$, **1.31 at $2$, 19.33 at $4$, 52.20 at $6$** —
which is exactly why the search variable is the fraction $u = c/c_\mathrm{max}(b)$
rather than $c$ itself: with $c$ free, the box would have to be sized for the
loosest $b$ and would be infeasible everywhere else.

`ForceConstantFit` ([fit/dissociation.py:1031](fit/dissociation.py#L1031)) reports
`terms`, `variables`, `scales` (absolute $c$), `decays`, `k_scales`, `depths`,
`curvatures`, and `margins` against `margins_before`. `BondVariable` is keyed
**per template**, not shared across templates — H–O in water and H–O in HO₂ are
separate variables.

**`fit.py` is not idempotent.** Re-running it over already-fitted output moves the
channel count on its own, so refit once from the previous state, and quote a
regression only against a baseline produced by the same pipeline over the same
inputs.

---

## 6. `EVBSystem` — the fixed-basis alternative

`evb.py:EVBSystem`, exposed as `ase.py:EVB` ([evb.py:13](evb.py#L13)). A fixed,
authored list of states; no network rebuild, no closure, no topology update, and
**no stress** — `EVB.implemented_properties` is `["energy", "forces"]` and the
virial returned by every force field is discarded at the call site
([evb.py:92](evb.py#L92)). It cannot drive a barostat.

$$
H_{ii} = E^{\text{bonded}}_i,
\qquad
H_{ij} = \sqrt{\big|(1+h)\,H_{ii}H_{jj}\big|} \;\ge\; 0,
$$

with hardness $h \in (0,1]$, default $0.95$. Gradient by the chain rule,

$$
\frac{\partial H_{ij}}{\partial \mathbf{x}}
= \operatorname{sgn}\big((1+h)H_{ii}H_{jj}\big)
\left[\frac{H_{ij}}{2H_{ii}}\frac{\partial H_{ii}}{\partial \mathbf{x}}
+ \frac{H_{ij}}{2H_{jj}}\frac{\partial H_{jj}}{\partial \mathbf{x}}\right],
$$

the sign factor being inert while $H_{ii}$ and $H_{jj}$ share a sign and wrong
when they do not.

**All three nonbonded terms go outside the Hamiltonian**, added once to the ground
state, because this coupling is a nonlinear function of the *absolute* diagonal
and a common shift does not pass through it. `ACKS2`, `ZBL` and the switched 12-6
are the same number on every state, so leaving them off the diagonal is exact.
`EVBSystem` finds the term dictionary for the two term-driven ones on the first
state that carries a non-empty one. Its results are `energy`, `forces`, the same
`energy_bonded` / `energy_nonbonded` / `energy_zbl` / `energy_lj` decomposition,
and `statevec` — which holds the *squared* amplitudes.

**The exclusions are where the two schemes now differ, and `EVBSystem` is the weaker
one.** Its ZBL and 12-6 exclusions are fine — they ride the per-state `term_dict`
through `bonded_ff` onto the diagonal, exactly as in `System`. The Coulomb screen
does not: `EVBSystem` calls `self.nonbonded_ff(pos, pbc, cell, state.term_dict)` on
the **first state carrying terms** ([evb.py:139](evb.py#L139)), so the screen is that
representative state's exclusions rather than the ground-state-weighted
$1 - \sum_s w_s M_s$ of §2.3.2. The comment there — "every state carries the same
`atom` terms, so state 0's term_dict is representative" — is true of the `atom`
terms and **false of the `coulombexclusion` terms**, which differ between states by
construction. The consequence is that `EVBSystem`'s total energy depends on the
order of its authored state list, which is the same class of defect
`test_nonbonded_is_independent_of_pivot` now rules out for `System`. The fix is to
build the screen from `statevecsq`; it has not been made, because `EVBSystem` holds
a fixed state list and is the simple alternative rather than the production path.

That placement used to be forced the *other* way for the Lennard-Jones: its
whole-system sum had to sit **on** the diagonal because the `exclusion` terms
cancelling it were already there, and splitting a cancelling pair across the two
sides left a water diagonal at $-1833$ eV against a bonded energy of $-9.87$,
with a coupling of $\sqrt{1.95\cdot 1833 \cdot 1829} = 2560$ eV following from
it. `ZBL` has no second half to be split from, so the question does not arise.

`System` also adds all three outside its Hamiltonian, for a different reason: its
couplings come from `EVBCoupling` and do not depend on the diagonal at all, so for
$H = D + V$ with $V$ fixed the placement there is free rather than forced. Here it
is forced, which is the one structural difference between the two schemes' energy
assembly.

---

## Source map

| Concept | Location |
| --- | --- |
| Calculator entry, topology feedback, stress | [ase.py:13](ase.py#L13) |
| Assembly, pivot, charge solve, screen, nonbonded addition | [system.py:103](system.py#L103) |
| Basis closure, switching, block Hamiltonian | [basis.py](basis.py) |
| Global parameters, defaults, manifest activation | [forcefield/params.py](forcefield/params.py) |
| Bonded terms, `_virial` | [forcefield/qforce.py](forcefield/qforce.py) |
| Screened-nuclear repulsion (tapered) | [forcefield/zbl.py](forcefield/zbl.py) |
| Switched 12-6, linear core, wall and dispersion | [forcefield/lj.py](forcefield/lj.py) |
| Charge equilibration, exclusion screen, `prepare`/`compute` | [forcefield/acks2.py](forcefield/acks2.py) |
| Charge kernel, minimum-image and Ewald, `contract` | [forcefield/ewald.py](forcefield/ewald.py) |
| Intramolecular exclusion derivation, and why Coulomb differs | [forcefield/exclusions.py](forcefield/exclusions.py) |
| RMSD coupling | [forcefield/coupling.py](forcefield/coupling.py) |
| Coupling fit ($A$, $a$) | [fit/coupling.py](fit/coupling.py) |
| Depth, bond-length and force-constant fit | [fit/dissociation.py](fit/dissociation.py) |
| Network, blocks, mappings | [core/reactionset.py](core/reactionset.py), [core/network.py](core/network.py), [core/reaction.py](core/reaction.py) |
| Gradient verification | `tests/test_gradients.py` |
| Virial / stress verification, symmetry, sign | `tests/test_stress.py` |
| Basis invariance, seed independence | `tests/test_evb_invariants.py` |
| NVE drift, $O(dt^2)$ check | `tests/test_energy_conservation.py` |
| Intermolecular collapse, the 1.6–2.0 Å gap | `tests/test_collapse.py` |
| Two halves of the LJ and ZBL decompositions agree | `tests/test_reference_energies.py`, `tests/test_gradients.py` |
| Periodic kernel against values; splitting independence, charged weight | `tests/test_ewald.py` |
| Dimer well, monomer charges, polarizability | `tests/test_water_structure.py` |
| Fit solves and bounds | `tests/test_fit.py` |
| Network fingerprint regression | `tests/test_get_network.py` |
| Cache correctness | `tests/test_optimizations.py` |
| Global parameters reach the force field; collisions refuse | `tests/test_params.py` |
