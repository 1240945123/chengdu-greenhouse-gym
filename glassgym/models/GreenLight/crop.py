from __future__ import annotations

import casadi as ca

from glassgym.models.GreenLight.aux_states import update


def crop_fluxes(x, u, d, p) -> dict[str, ca.SX]:
    """Return the Vanthoor crop carbon fluxes used by GreenLight."""
    auxiliary = update(x, u, d, p)
    return {
        "photosynthesis": auxiliary[200],
        "buffer_to_leaf": auxiliary[206],
        "buffer_to_stem": auxiliary[207],
        "buffer_to_fruit": auxiliary[208],
        "growth_respiration": auxiliary[209],
        "leaf_maintenance": auxiliary[210],
        "stem_maintenance": auxiliary[211],
        "fruit_maintenance": auxiliary[212],
        "leaf_pruning": auxiliary[214],
        "fruit_harvest": auxiliary[215],
    }


def crop_derivatives(x, u, d, p) -> ca.SX:
    """Compute derivatives for cBuf, cLeaf, cStem, cFruit, tCanSum and time."""
    flux = crop_fluxes(x, u, d, p)
    derivatives = ca.vertcat(
        flux["photosynthesis"]
        - flux["buffer_to_fruit"]
        - flux["buffer_to_leaf"]
        - flux["buffer_to_stem"]
        - flux["growth_respiration"],
        flux["buffer_to_leaf"] - flux["leaf_maintenance"] - flux["leaf_pruning"],
        flux["buffer_to_stem"] - flux["stem_maintenance"],
        flux["buffer_to_fruit"] - flux["fruit_maintenance"] - flux["fruit_harvest"],
        x[4] / 86400.0,
        1.0 / 86400.0,
    )
    return derivatives
