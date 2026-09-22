"""Which electrostatic term a dataset uses: `params.ForceFieldParams.electrostatics`.

Resolved at call time, like every other global parameter, because the fitter
builds its force fields at import -- before any manifest has said which one it
was fitted with.
"""

from __future__ import annotations

from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.params import ForceFieldParams, resolve
from DynamicTopology.forcefield.pointcharge import PointCharge

_CLASSES = {"acks2": ACKS2, "pointcharge": PointCharge}


class Electrostatics:
    """One instance of each term, handing out whichever the active set names.

    Each instance carries a per-cell Ewald cache, so switching between them must
    not rebuild it; keeping one of each is what `System`, `EVBSystem` and the
    fitter all want.
    """

    def __init__(self):
        self._instances: dict[str, object] = {}

    def get(self, params: ForceFieldParams | None = None):
        name = resolve(params).electrostatics
        instance = self._instances.get(name)
        if instance is None:
            instance = self._instances[name] = _CLASSES[name]()
        return instance

    def __call__(self, pos, pbc, cell, term_dict: dict):
        return self.get()(pos, pbc, cell, term_dict)
