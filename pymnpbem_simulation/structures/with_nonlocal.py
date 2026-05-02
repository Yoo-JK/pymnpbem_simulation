from typing import Any, Dict, Tuple

import numpy as np

from .base import StructureBuilder
from ..util import print_info


class WithNonlocalBuilder(StructureBuilder):
    """기존 구조에 nonlocal hydrodynamic Drude eps 를 적용 (EpsFun 우회).

    YAML config::

        structure:
          type: with_nonlocal
          base:
            type: sphere
            diameter: 30
            mesh_density: 60
          nonlocal:
            metal: gold              # gold / silver / aluminum / *.dat
            beta: null               # m/s. null = sqrt(3/5)*vF default
            k_nm_inv: null            # null 또는 0.0 → local limit (= base 와 동일)
                                     # >0 → 그 wavenumber 에서의 nonlocal eps

    동작:
      1. base 구조를 빌드해 (p, epstab, n) 획득
      2. epstab[1] (particle eps) 를 nonlocal EpsFun 으로 교체
      3. p.eps / p.pc.eps 도 동기화 (다운스트림 simulation runner 가 참조)

    EpsNonlocal 정식 port 가 추가되기 전까지의 우회 wrapper. k=0 한정 동작이며,
    공간 의존 nonlocal 효과는 EpsNonlocal 클래스가 들어와야 정확.
    """

    def build(self) -> Tuple[Any, Any, int]:
        from . import build_structure
        from ..material import build_nonlocal_eps

        cfg_base = self.cfg_struct.get('base', None)
        if cfg_base is None:
            raise ValueError('[error] <structure.base> required for type=with_nonlocal')

        cfg_nl = self.cfg_struct.get('nonlocal', dict())

        nl_spec = {
                'type': 'nonlocal',
                'base': cfg_nl.get('metal', cfg_nl.get('base', 'gold')),
                'beta': cfg_nl.get('beta', None),
                'k_nm_inv': cfg_nl.get('k_nm_inv', None)}

        eps_nonlocal = build_nonlocal_eps(nl_spec)

        p, epstab_base, nfaces = build_structure(cfg_base, self.cfg_materials)

        epstab = list(epstab_base)
        if len(epstab) < 2:
            raise RuntimeError(
                    '[error] base structure produced epstab len={} < 2'.format(len(epstab)))

        # Replace the particle eps slot (index 1, MATLAB-1based = idx 2)
        epstab[1] = eps_nonlocal

        try:
            p.eps = list(epstab)
        except Exception:
            pass

        if hasattr(p, 'pc') and p.pc is not None:
            try:
                p.pc.eps = list(epstab)
            except Exception:
                pass

        print_info('WithNonlocal: base={}, metal={}, beta={} m/s, k_nm_inv={}'.format(
                cfg_base.get('type'), nl_spec['base'], nl_spec['beta'], nl_spec['k_nm_inv']))
        print_info('WithNonlocal: nfaces={}, eps[1] -> EpsFun(nonlocal Drude)'.format(nfaces))

        return p, epstab, nfaces
