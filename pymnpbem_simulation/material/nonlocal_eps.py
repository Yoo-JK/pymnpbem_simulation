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
        metal: gold        # 'gold'|'silver'|'aluminum'|'copper'|'from_table:<path>.dat'
        beta: null         # eV*nm. null -> sqrt(3/5) * v_F * hbar default for metal
        delta_d: 0.05      # nm — artificial cover-layer thickness
        eps_embed: 1.0     # outer medium permittivity (scalar) or '<path>.dat'
        drude_params:      # optional explicit Drude override
          omega_p: 9.02
          gamma: 0.071
          eps_inf: 9.84
          v_f: 1.39e6

EpsFun-based legacy wrapper (M7 Wave 3) is removed; this module now
delegates to the canonical EpsNonlocal class.
"""

from typing import Any, Dict, Tuple

import numpy as np

from mnpbem.materials import EpsConst, EpsTable, EpsDrude, EpsNonlocal, make_nonlocal_pair


_DEFAULT_DELTA_D_NM = 0.05

_HBAR_EV_S = 6.582119569e-16

# Drude parameters used when the metal has no canonical mnpbem EpsDrude
# factory (copper) or when the caller asks for an explicit override.
# omega_p / gamma / eps_inf in eV, v_f in m/s.
_DRUDE_PRESETS = {
    'gold': {'omega_p': 9.02, 'gamma': 0.071, 'eps_inf': 9.84, 'v_f': 1.39e6},
    'silver': {'omega_p': 9.17, 'gamma': 0.021, 'eps_inf': 3.7, 'v_f': 1.39e6},
    'aluminum': {'omega_p': 14.98, 'gamma': 0.047, 'eps_inf': 1.0, 'v_f': 2.03e6},
    'copper': {'omega_p': 10.83, 'gamma': 0.073, 'eps_inf': 1.0, 'v_f': 1.57e6},
}

_METAL_ALIASES = {
    'au': 'gold', 'gold': 'gold',
    'ag': 'silver', 'silver': 'silver',
    'al': 'aluminum', 'aluminum': 'aluminum', 'aluminium': 'aluminum',
    'cu': 'copper', 'copper': 'copper',
}

# Metals that mnpbem's make_nonlocal_pair / EpsDrude can build directly.
_MNPBEM_NATIVE_METALS = {'au', 'gold', 'ag', 'silver', 'al', 'aluminum', 'aluminium'}


def canonical_metal(name: Any) -> str:
    return _METAL_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())


def beta_from_fermi_velocity(v_f: float) -> float:
    return _HBAR_EV_S * np.sqrt(3.0 / 5.0) * float(v_f) * 1.0e9


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
    model = str(spec.get('model', 'hydrodynamic')).strip().lower()

    if model not in {'hydrodynamic', 'qcm'}:
        raise ValueError(
                '[error] build_nonlocal_eps: unknown <model>=<{}> '
                '(expected hydrodynamic or qcm)'.format(model))

    if model == 'qcm':
        # Matches the MATLAB wrapper, which also records 'qcm' but only ever
        # evaluates the hydrodynamic cover layer.
        print('[warn] build_nonlocal_eps: <model: qcm> is not implemented; '
                'falling back to the hydrodynamic cover layer')

    if eps_embed is None:
        eps_embed = _resolve_eps_embed(spec.get('eps_embed', 1.0))

    drude_params = _resolve_drude_params(spec.get('drude_params', None), metal_l)

    # Branch O: explicit Drude override, or a metal mnpbem cannot build itself
    # (copper). Both need EpsNonlocal wired up by hand.
    if drude_params is not None or (metal_l not in _MNPBEM_NATIVE_METALS
            and canonical_metal(metal_l) in _DRUDE_PRESETS):
        params = drude_params if drude_params is not None else _preset_for(metal_l)
        eps_metal_core = _core_eps_from(metal, metal_l, params)
        if beta is None and params.get('v_f', None) is not None:
            beta = beta_from_fermi_velocity(params['v_f'])
        eps_shell = EpsNonlocal(eps_metal_core, eps_embed,
                delta_d = delta_d,
                eps_inf = params['eps_inf'],
                omega_p = params['omega_p'],
                gamma = params['gamma'],
                beta = beta,
                name = metal)
        return eps_embed, eps_metal_core, eps_shell

    # Branch A: built-in metal name -> use make_nonlocal_pair helper.
    if metal_l in _MNPBEM_NATIVE_METALS:
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
            '(expected gold/silver/aluminum/copper or from_table:<path>.dat)'.format(metal))


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
    if n in {'copper', 'cu'}:
        p = _DRUDE_PRESETS['copper']
        return lambda: EpsDrude(p['eps_inf'], p['omega_p'], p['gamma'], name = 'Cu')
    raise ValueError(
            '[error] build_nonlocal_eps: no Drude factory for <{}>'.format(name))


def _preset_for(metal: str) -> Dict[str, Any]:
    key = canonical_metal(metal)
    if key not in _DRUDE_PRESETS:
        raise ValueError(
                '[error] build_nonlocal_eps: no Drude preset for <{}> '
                '(known: {})'.format(metal, sorted(_DRUDE_PRESETS)))
    return dict(_DRUDE_PRESETS[key])


def _resolve_drude_params(raw: Any, metal: str) -> Any:
    # Accepts the flat {'omega_p': ...} form and the MATLAB-wrapper form keyed
    # by material name, {'gold': {'omega_p': ...}}. None -> canonical mnpbem path.
    if not isinstance(raw, dict) or not raw:
        return None

    entry = raw

    # Material-keyed form: pick the entry matching this metal.
    if not any(k in raw for k in ('omega_p', 'gamma', 'eps_inf', 'v_f')):
        key = canonical_metal(metal)
        match = None
        for name, value in raw.items():
            if canonical_metal(name) == key and isinstance(value, dict):
                match = value
                break
        if match is None:
            return None
        entry = match

    params = _preset_for(metal) if canonical_metal(metal) in _DRUDE_PRESETS else dict()

    for key in ('omega_p', 'gamma', 'eps_inf', 'v_f'):
        if entry.get(key, None) is not None:
            params[key] = float(entry[key])

    missing = [k for k in ('omega_p', 'gamma', 'eps_inf') if k not in params]
    if missing:
        raise ValueError(
                '[error] build_nonlocal_eps: <drude_params> for <{}> is missing '
                '{}'.format(metal, missing))

    return params


def _core_eps_from(metal: str, metal_l: str, params: Dict[str, Any]) -> Any:
    if metal_l.startswith('from_table:'):
        path = metal.split(':', 1)[1].strip()
        if not path:
            raise ValueError(
                    '[error] build_nonlocal_eps: <from_table:> requires a path')
        return EpsTable(path)

    if metal.endswith('.dat'):
        return EpsTable(metal)

    return EpsDrude(params['eps_inf'], params['omega_p'], params['gamma'],
            name = metal)
