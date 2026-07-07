from typing import Any, Dict, Tuple, List

import numpy as np

from .base import StructureBuilder
from ..util import print_info


_VALID_MIRROR_KEYS = {'x', 'y', 'xy'}


class WithMirrorBuilder(StructureBuilder):
    """Apply mirror symmetry to an existing structure so only half the mesh is solved by BEM.
    (기존 구조에 mirror symmetry 를 적용해 절반 mesh 만 BEM 으로 푼다.)

    YAML config::

        structure:
          type: with_mirror
          base:
            type: sphere
            diameter: 30
            mesh_density: 60
          mirror:
            sym: xy           # 'x' / 'y' / 'xy' (same as MNPBEM ComParticleMirror / 와 동일)

    Flow (동작 흐름):
      1. Build the ``base`` structure via the existing REGISTRY (``p_base, epstab, _``).
         (``base`` 구조를 기존 REGISTRY 로 빌드.)
      2. Extract ``ComParticle`` inout/particles info and rebuild as ``ComParticleMirror``.
         (``ComParticle`` 의 inout/particles 를 추출해 ``ComParticleMirror`` 로 재구성.)
      3. Return the mirror particle as-is; the downstream simulation runner must auto-dispatch
         to a BEM*Mirror solver instead of the plain BEMRet/BEMStat.
         (mirror 입자를 반환. 다운스트림 runner 는 자동 dispatch 로 BEM*Mirror solver 선택.)

    Meaning of sym (MATLAB equivalent / MATLAB 동등):
      - 'x'  : x=0 mirror plane → 2x total particle count; base mesh stays the original.
               (x=0 mirror plane → 전체 입자 수 2 배. base mesh 는 원본 그대로.)
      - 'y'  : y=0 mirror plane → same. (동일.)
      - 'xy' : x=0, y=0 two planes → 4x total particle count; 4x speedup.
               (두 plane → 전체 입자 수 4 배. 4× 가속.)
    """

    def build(self) -> Tuple[Any, Any, int]:
        from . import build_structure

        cfg_base = self.cfg_struct.get('base', None)

        if cfg_base is None:
            raise ValueError(
                '[error] <structure.base> required for type=with_mirror')

        cfg_mirror = self.cfg_struct.get('mirror', dict())
        sym = str(cfg_mirror.get('sym', 'xy')).lower()

        if sym not in _VALID_MIRROR_KEYS:
            raise ValueError(
                '[error] <mirror.sym> must be one of {}, got <{}>'.format(
                        sorted(_VALID_MIRROR_KEYS), sym))

        p_base, epstab, nfaces_base = build_structure(cfg_base, self.cfg_materials)

        particles = _extract_particle_list(p_base)
        inout = _extract_inout(p_base, n_particles = len(particles))

        from mnpbem.geometry import ComParticleMirror

        p_mirror = ComParticleMirror(epstab, particles, inout, sym = sym)

        nfaces_full = int(p_mirror.pfull.nfaces) if hasattr(p_mirror, 'pfull') else nfaces_base
        nfaces_half = int(p_mirror.nfaces)

        try:
            setattr(p_mirror, '_mnpbem_mirror_sym', sym)
        except Exception:
            pass

        print_info(
                'WithMirror: base={}, sym={}, nfaces_full={}, nfaces_half={}'.format(
                        cfg_base.get('type'), sym, nfaces_full, nfaces_half))

        return p_mirror, epstab, nfaces_half


def _extract_particle_list(p_base: Any) -> List[Any]:
    if hasattr(p_base, 'p') and isinstance(p_base.p, (list, tuple)):
        return list(p_base.p)

    if hasattr(p_base, 'pfull') and hasattr(p_base.pfull, 'p'):
        return list(p_base.pfull.p)

    raise AttributeError(
            '[error] cannot extract particle list from <{}>'.format(type(p_base).__name__))


def _extract_inout(p_base: Any,
        n_particles: int) -> np.ndarray:
    inout = getattr(p_base, 'inout', None)

    if inout is None and hasattr(p_base, 'pfull'):
        inout = getattr(p_base.pfull, 'inout', None)

    if inout is None:
        raise AttributeError(
                '[error] cannot extract inout from base particle')

    inout = np.atleast_2d(np.asarray(inout))

    if inout.shape[0] != n_particles:
        if inout.shape[1] == n_particles and inout.shape[0] != n_particles:
            inout = inout.T

    return inout
