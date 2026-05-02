from typing import Any, Dict, Optional, Callable, Union, Tuple

import numpy as np

from mnpbem.materials import EpsFun, EpsTable, EpsConst, EpsDrude


# eV ↔ nm 변환 상수 (MNPBEM 와 동일: 1 / 8.0655477e-4)
_EV2NM = 1.0 / 8.0655477e-4

# Fermi velocity (m/s) for selected metals — Boardman / Mortensen 등에서 사용하는 값
# beta = sqrt(3/5) * vF  (hydrodynamic Drude longitudinal sound speed)
# 기본 단위 m/s; nm/s 로 환산 필요할 때 _M_PER_S_TO_NM_PER_S 곱한다.
_M_PER_S_TO_NM_PER_S = 1.0e9

_FERMI_VELOCITY_M_S = {
        'au':       1.40e6,
        'gold':     1.40e6,
        'ag':       1.39e6,
        'silver':   1.39e6,
        'al':       2.03e6,
        'aluminum': 2.03e6,
        'aluminium':2.03e6}


def is_nonlocal_spec(spec: Any) -> bool:
    if isinstance(spec, dict):
        t = str(spec.get('type', '')).lower()
        return t in {'nonlocal', 'hydrodynamic', 'nonlocal_drude'}
    return False


def build_nonlocal_eps(spec: Dict[str, Any]) -> EpsFun:
    """Nonlocal hydrodynamic eps 를 EpsFun 으로 감싸 반환.

    spec 예시 (YAML 에서 그대로 변환됨)::

        material:
          type: nonlocal           # or 'hydrodynamic'
          base: gold               # 'gold' / 'silver' / 'aluminum' / '*.dat'
          beta: null               # m/s. null 이면 sqrt(3/5)*vF default
          k_nm_inv: null           # null = local limit (k=0)
                                   # 숫자값이면 그 wavenumber 에서의 nonlocal eps

    Hydrodynamic Drude formula:
        eps(omega, k) = eps_inf - omega_p^2 / (omega^2 + i * gamma * omega - beta^2 * k^2)

    base eps 가 EpsTable (e.g. gold.dat) 이면 local 곡선을 그대로 쓰고
    오직 longitudinal 보정만 추가한다 (eps_table - eps_drude_local + eps_drude_nonlocal).
    base eps 가 EpsDrude 이면 위 공식 직접 사용.

    EpsFun 시그니처는 ``f(enei) -> eps_complex`` 이므로 EpsNonlocal 정식 port 와 달리
    공간 의존 (k) 정보를 전달할 수 없다. 따라서 wrapper 는 k 를 spec 에서 한 번 받아
    "그 k 에서의 dispersion-corrected eps" 만 반환한다. k=0 이면 local 결과와 정확히 일치.
    """

    if not is_nonlocal_spec(spec):
        raise ValueError(
                '[error] build_nonlocal_eps: spec is not a nonlocal type: <{}>'.format(spec))

    base_name = str(spec.get('base', 'gold')).lower()
    beta_m_s = spec.get('beta', None)
    k_nm_inv = spec.get('k_nm_inv', None)

    if beta_m_s is None:
        vF = _FERMI_VELOCITY_M_S.get(base_name, None)
        if vF is None:
            raise ValueError(
                    '[error] no default Fermi velocity for <{}>; specify <beta> in m/s'.format(
                            base_name))
        beta_m_s = float(np.sqrt(3.0 / 5.0) * vF)

    beta_nm_per_s = float(beta_m_s) * _M_PER_S_TO_NM_PER_S

    base_eps = _resolve_base_eps(base_name)

    if k_nm_inv is None or float(k_nm_inv) == 0.0:
        return EpsFun(_make_local_passthrough(base_eps), key = 'nm')

    k = float(k_nm_inv)

    return EpsFun(_make_hydrodynamic_corrected(base_eps, base_name, beta_nm_per_s, k),
            key = 'nm')


def make_hydrodynamic_drude_eps(eps_inf: float,
        wp_eV: float,
        gamma_eV: float,
        beta_m_s: float,
        k_nm_inv: float) -> EpsFun:
    """순수 hydrodynamic Drude eps wrapper. base 에 표 데이터를 쓰지 않을 때.

    eps(omega, k) = eps_inf - wp^2 / (omega^2 + i*gamma*omega - beta_eV^2 * k^2)
    """
    beta_nm_per_s = float(beta_m_s) * _M_PER_S_TO_NM_PER_S

    def _f(enei: np.ndarray) -> np.ndarray:
        omega_eV = _EV2NM / np.asarray(enei, dtype = float)
        # angular frequency in rad/s
        # MATLAB MNPBEM 은 eps 식을 eV 단위로 푼다. beta * k 도 eV 로 환산하려면
        # hbar/c (eV·nm) 단위 수표 사용. 여기서는 hydrodynamic correction 을
        # eV 단위로 환산: beta_eV = (hbar * beta_nm_per_s) where hbar in eV*s = 6.582e-16
        hbar_eVs = 6.582119569e-16
        beta_eV_nm = hbar_eVs * beta_nm_per_s   # eV·nm

        denom = omega_eV * (omega_eV + 1j * gamma_eV) - (beta_eV_nm * k_nm_inv) ** 2
        eps_val = eps_inf - (wp_eV ** 2) / denom
        return np.asarray(eps_val, dtype = complex)

    return EpsFun(_f, key = 'nm')


def _resolve_base_eps(name: str) -> Any:
    if name in {'gold', 'au'}:
        return EpsTable('gold.dat')
    if name in {'silver', 'ag'}:
        return EpsTable('silver.dat')
    if name in {'aluminum', 'aluminium', 'al'}:
        return EpsDrude.aluminum()
    if name.endswith('.dat'):
        return EpsTable(name)

    raise ValueError(
            '[error] Unsupported nonlocal <base>=<{}>'.format(name))


def _make_local_passthrough(base_eps: Any) -> Callable[[np.ndarray], np.ndarray]:

    def _f(enei: np.ndarray) -> np.ndarray:
        eps, _k = base_eps(np.asarray(enei, dtype = float))
        return np.asarray(eps, dtype = complex)

    return _f


def _make_hydrodynamic_corrected(base_eps: Any,
        base_name: str,
        beta_nm_per_s: float,
        k_nm_inv: float) -> Callable[[np.ndarray], np.ndarray]:
    """k>0 일 때 hydrodynamic longitudinal correction 추가.

    근사 흐름:
      1. base_eps(enei) -> eps_local
      2. EpsDrude (해당 metal) 의 eps_local_drude 추출
      3. eps_nonlocal_drude = eps_inf - wp^2 / (w(w+ig) - beta^2 k^2)
      4. result = eps_local - eps_local_drude + eps_nonlocal_drude
         (= 인터밴드 + 보정된 Drude)

    base 가 EpsDrude 이면 eps_local_drude == eps_local 이므로 단순히 nonlocal Drude 만 반환.
    base 가 EpsTable 이면 인터밴드는 그대로 두고 Drude 부분만 nonlocal 로 갈음한다.
    """
    drude_metal = _drude_for(base_name)

    hbar_eVs = 6.582119569e-16
    beta_eV_nm = hbar_eVs * beta_nm_per_s

    def _f(enei: np.ndarray) -> np.ndarray:
        enei_arr = np.asarray(enei, dtype = float)
        eps_local, _k = base_eps(enei_arr)
        eps_local = np.asarray(eps_local, dtype = complex)

        # local Drude piece (subtract this if base is a tabulated table)
        eps_drude_local, _kd = drude_metal(enei_arr)
        eps_drude_local = np.asarray(eps_drude_local, dtype = complex)

        omega_eV = _EV2NM / enei_arr
        wp = drude_metal.wp
        gamma = drude_metal.gammad
        eps_inf = drude_metal.eps0

        denom = omega_eV * (omega_eV + 1j * gamma) - (beta_eV_nm * k_nm_inv) ** 2
        eps_drude_nonlocal = eps_inf - (wp ** 2) / denom

        if isinstance(base_eps, EpsTable):
            return eps_local - eps_drude_local + eps_drude_nonlocal
        # base is EpsDrude or EpsConst — overwrite Drude part entirely
        return eps_drude_nonlocal

    return _f


def _drude_for(name: str) -> EpsDrude:
    if name in {'gold', 'au'}:
        return EpsDrude.gold()
    if name in {'silver', 'ag'}:
        return EpsDrude.silver()
    if name in {'aluminum', 'aluminium', 'al'}:
        return EpsDrude.aluminum()
    raise ValueError(
            '[error] no Drude reference for nonlocal base <{}>'.format(name))
