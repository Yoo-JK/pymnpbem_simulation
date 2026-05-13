"""Particle wrapped with a nonlocal hydrodynamic Drude cover layer.

Builds a 2-layer ComParticle following the Yu Luo / Pendry et al. (PRL 111,
093901, 2013) effective-cover-layer formulation, matching MATLAB
demospecstat19.m::

    epstab    = [eps_embed, eps_metal_core, eps_nonlocal_shell]
    particles = [p_shell, p_core]
    inout     = [[3, 1], [2, 3]]      # 1-based indices into epstab
    p         = ComParticle(epstab, particles, inout, 1, 2)
    refun     = coverlayer.refine(p, [[1, 2]])
    bem       = BEMStat(p, refun = refun)

The base sub-structure config defines the *inner* (Drude metal) geometry.
The shell with thickness ``delta_d`` is created via
``coverlayer.shift(p_core, delta_d)``.

Example YAML::

    structure:
      type: with_nonlocal
      base:
        type: sphere
        diameter: 10
        mesh_density: 144
        interp: curv
      nonlocal:
        metal: gold
        delta_d: 0.05
        beta: null            # default sqrt(3/5)*v_F*hbar
        eps_embed: 1.0

The refun callable is attached on the returned ComParticle as
``_mnpbem_refun`` (analogous to WithSubstrateBuilder's ``_mnpbem_layer``).
SimulationRunner subclasses pick it up when constructing the BEM solver.
"""

from typing import Any, Dict, Tuple

from .base import StructureBuilder
from ..util import print_info


class WithNonlocalBuilder(StructureBuilder):
    """Wrap base structure with a nonlocal hydrodynamic Drude cover layer."""

    def build(self) -> Tuple[Any, Any, int]:
        from mnpbem.geometry import ComParticle
        from mnpbem.greenfun import coverlayer

        from . import build_structure
        from ..material.nonlocal_eps import build_nonlocal_eps

        cfg_base = self.cfg_struct.get('base', None)
        if cfg_base is None:
            raise ValueError('[error] <structure.base> required for type=with_nonlocal')

        cfg_nl = self.cfg_struct.get('nonlocal', dict())

        nl_spec = dict(cfg_nl)
        nl_spec.setdefault('type', 'nonlocal')
        nl_spec.setdefault('metal', cfg_nl.get('metal', cfg_nl.get('base', 'gold')))

        # 1) Build base ComParticle, then extract the inner Particle objects.
        p_base, _epstab_base, _nfaces_base = build_structure(cfg_base, self.cfg_materials)

        if not hasattr(p_base, 'p') or len(p_base.p) == 0:
            raise RuntimeError(
                    '[error] WithNonlocal: base structure did not expose Particle list')

        # 2) Build (eps_embed, eps_core, eps_shell). eps_embed defaults to
        #    EpsConst(1.0) if not specified in nonlocal.eps_embed.
        eps_embed, eps_metal_core, eps_shell = build_nonlocal_eps(nl_spec)

        delta_d = float(eps_shell.delta_d)

        # 3) Per-sub-particle: shift each base particle outward by delta_d to
        #    form the artificial cover-layer shell. Then assemble flat lists
        #    [shell_1, core_1, shell_2, core_2, ...] together with epstab and
        #    inout matching MATLAB's [3, 1; 2, 3] pattern (replicated per
        #    sub-particle).
        particles_out = []
        inout_rows = []
        for p_core in p_base.p:
            p_shell = coverlayer.shift(p_core, delta_d)
            particles_out.append(p_shell)
            particles_out.append(p_core)
            # epstab indices (1-based): 1=embed, 2=core_drude, 3=nonlocal_shell.
            # Shell row [3, 1]: inside=shell, outside=embed.
            # Core row  [2, 3]: inside=core,  outside=shell.
            inout_rows.append([3, 1])
            inout_rows.append([2, 3])

        epstab = [eps_embed, eps_metal_core, eps_shell]

        # closed = list(1..n_sub) in MATLAB sense — every sub-particle here is
        # a closed surface (shell wraps the core, core itself is closed).
        n_sub = len(particles_out)
        closed_args = list(range(1, n_sub + 1))

        # 4) Carry over interp setting from the base config.
        interp = cfg_base.get('interp', 'curv')

        p = ComParticle(epstab, particles_out, inout_rows, *closed_args,
                interp = interp)

        # 5) Build refun for polar-integration refinement of cover-layer pairs.
        #    For each shell/core pair, MATLAB passes [shell_idx, core_idx];
        #    sub-particles in our flat list are 1-based: shell at 2k+1, core
        #    at 2k+2 (k = 0..n_base-1).
        refun_pairs = []
        for k in range(len(p_base.p)):
            shell_idx = 2 * k + 1
            core_idx = 2 * k + 2
            refun_pairs.append([shell_idx, core_idx])

        refun = coverlayer.refine(p, refun_pairs)

        # 6) Stash refun on the particle so SimulationRunner can forward it
        #    to BEMStat via the **options channel (CompGreenStat accepts
        #    ``refun=...``).
        try:
            setattr(p, '_mnpbem_refun', refun)
        except Exception:
            if hasattr(p, 'pfull'):
                setattr(p.pfull, '_mnpbem_refun', refun)

        # 7) face count
        nfaces = _count_faces(p)

        print_info('WithNonlocal: base={}, metal={}, delta_d={}nm, beta={} eV*nm'.format(
                cfg_base.get('type'),
                nl_spec.get('metal'),
                delta_d,
                getattr(eps_shell, 'beta', None)))
        print_info('WithNonlocal: epstab len=3 [embed, core, shell], n_sub={}, nfaces={}'.format(
                n_sub, nfaces))

        return p, epstab, nfaces


def _count_faces(p: Any) -> int:
    if hasattr(p, 'pfull') and p.pfull is not None:
        return int(getattr(p.pfull, 'nfaces', -1))
    if hasattr(p, 'nfaces'):
        return int(p.nfaces)
    if hasattr(p, 'pos'):
        return int(len(p.pos))
    return -1
