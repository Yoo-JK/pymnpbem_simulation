"""Nonlocal hydrodynamic Drude eps via mnpbem.materials.EpsNonlocal.

Yu Luo / Pendry et al. (PRL 111, 093901, 2013) artificial cover-layer
formulation. Wraps the canonical EpsNonlocal port from mnpbem so YAML
config consumers can build the (core_metal_eps, nonlocal_shell_eps) pair
plus the embedding eps in one call.

YAML schema (consumed by `WithNonlocalBuilder`)::

    structure:
      type: with_nonlocal
      base:
        type: sphere
        diameter: 10
        mesh_density: 144
      nonlocal:
        metal: gold        # 'gold' | 'silver' | 'aluminum' | 'from_table:<path>.dat'
        beta: null         # eV*nm. null -> sqrt(3/5) * v_F * hbar default for metal
        delta_d: 0.05      # nm — artificial cover-layer thickness
        eps_embed: 1.0     # outer medium permittivity (scalar) or '<path>.dat'

EpsFun-based legacy wrapper (M7 Wave 3) is removed; this module now
delegates to the canonical EpsNonlocal class.
"""

from typing import Any, Dict, Tuple

from mnpbem.materials import EpsConst, EpsTable, EpsDrude, EpsNonlocal, make_nonlocal_pair


_DEFAULT_DELTA_D_NM = 0.05


def is_nonlocal_spec(spec: Any) -> bool:
    if isinstance(spec, dict):
        t = str(spec.get('type', '')).lower()
        return t in {'nonlocal', 'hydrodynamic', 'nonlocal_drude'}
    return False


def build_nonlocal_eps(spec: Dict[str, Any],
        eps_embed: Any = None) -> Tuple[Any, Any, Any]:
    """YAML config nonlocal entry -> (eps_embed, eps_metal_core, eps_shell).

    Parameters
    ----------
    spec : dict
        Nonlocal spec block. Recognized keys:
          - ``metal`` (str, default 'gold'): 'gold' / 'silver' / 'aluminum'
            or 'from_table:<path>.dat' (Johnson-Christy etc.).
          - ``beta`` (float or None): hydrodynamic velocity in eV*nm. If
            None, defaults from sqrt(3/5)*v_F*hbar for the metal.
          - ``delta_d`` (float, default 0.05): cover-layer thickness in nm.
          - ``eps_embed`` (float or str): only used if ``eps_embed`` arg is
            None; 1.0 -> EpsConst(1.0); string ending in .dat -> EpsTable.
    eps_embed : EpsConst-like, optional
        Outer medium dielectric. If None, built from spec.

    Returns
    -------
    eps_embed : EpsConst / EpsTable
        Outer-medium permittivity (epstab[0]).
    eps_metal_core : EpsDrude or EpsTable
        Local Drude (or tabulated) permittivity of the inner metal core
        (epstab[1]).
    eps_shell : EpsNonlocal
        Artificial thin-cover-layer permittivity (epstab[2]).

    Notes
    -----
    The downstream geometry is a 2-layer ComParticle::

        epstab    = [eps_embed, eps_metal_core, eps_shell]
        particles = [p_shell, p_core]
        inout     = [[3, 1], [2, 3]]      # MATLAB demospecstat19.m

    See WithNonlocalBuilder for the full setup.
    """

    if not isinstance(spec, dict):
        raise ValueError(
                '[error] build_nonlocal_eps: spec must be a dict, got <{}>'.format(type(spec).__name__))

    metal_raw = spec.get('metal', spec.get('base', 'gold'))
    if metal_raw is None:
        raise ValueError('[error] build_nonlocal_eps: <metal> is None')

    metal = str(metal_raw).strip()
    metal_l = metal.lower()

    beta = spec.get('beta', None)
    delta_d = float(spec.get('delta_d', _DEFAULT_DELTA_D_NM))

    if eps_embed is None:
        eps_embed = _resolve_eps_embed(spec.get('eps_embed', 1.0))

    # Branch A: built-in metal name -> use make_nonlocal_pair helper.
    if metal_l in {'au', 'gold', 'ag', 'silver', 'al', 'aluminum', 'aluminium'}:
        eps_metal_core, eps_shell = make_nonlocal_pair(metal_l,
                eps_embed = eps_embed,
                delta_d = delta_d,
                beta = beta)
        return eps_embed, eps_metal_core, eps_shell

    # Branch B: 'from_table:<path>' -> tabulated metal core, Drude params from
    # canonical EpsDrude.<drude_metal>() (default gold) for longitudinal correction.
    if metal_l.startswith('from_table:'):
        path = metal.split(':', 1)[1].strip()
        if not path:
            raise ValueError(
                    '[error] build_nonlocal_eps: <from_table:> requires a path')
        drude_metal = spec.get('drude_metal', 'gold')
        drude_factory = _drude_factory_for(drude_metal)
        eps_drude = drude_factory()
        eps_table = EpsTable(path)
        eps_shell = EpsNonlocal(eps_table, eps_embed,
                delta_d = delta_d,
                eps_inf = eps_drude.eps0,
                omega_p = eps_drude.wp,
                gamma = eps_drude.gammad,
                beta = beta,
                name = eps_drude.name)
        return eps_embed, eps_table, eps_shell

    # Branch C: plain '<file>.dat' -> tabulated metal with gold Drude params.
    if metal.endswith('.dat'):
        eps_drude = EpsDrude.gold()
        eps_table = EpsTable(metal)
        eps_shell = EpsNonlocal(eps_table, eps_embed,
                delta_d = delta_d,
                eps_inf = eps_drude.eps0,
                omega_p = eps_drude.wp,
                gamma = eps_drude.gammad,
                beta = beta,
                name = 'Au')
        return eps_embed, eps_table, eps_shell

    raise ValueError(
            '[error] build_nonlocal_eps: unknown <metal>=<{}> '
            '(expected gold/silver/aluminum or from_table:<path>.dat)'.format(metal))


def _resolve_eps_embed(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return EpsConst(float(value))
    if isinstance(value, str):
        if value.endswith('.dat'):
            return EpsTable(value)
        try:
            return EpsConst(float(value))
        except ValueError:
            raise ValueError(
                    '[error] build_nonlocal_eps: cannot resolve <eps_embed>=<{}>'.format(value))
    if hasattr(value, '__call__'):
        return value
    raise ValueError(
            '[error] build_nonlocal_eps: invalid <eps_embed>=<{}>'.format(value))


def _drude_factory_for(name: str) -> Any:
    n = str(name).lower()
    if n in {'gold', 'au'}:
        return EpsDrude.gold
    if n in {'silver', 'ag'}:
        return EpsDrude.silver
    if n in {'aluminum', 'aluminium', 'al'}:
        return EpsDrude.aluminum
    raise ValueError(
            '[error] build_nonlocal_eps: no Drude factory for <{}>'.format(name))
