"""User material example: gold-like Drude permittivity.

Factory name is intentionally `generate_eps_func` to match the default
descriptor loader expectation.
"""

from mnpbem.materials import EpsDrude


def generate_eps_func():
    """Return gold-like Drude dielectric function (EpsDrude)."""
    return EpsDrude(eps0 = 9.54, wp = 9.03, gammad = 0.071)
