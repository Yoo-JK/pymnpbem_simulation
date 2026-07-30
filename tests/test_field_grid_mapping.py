"""필드 배열 -> 격자 매핑 회귀 테스트.

rod 캠페인에서 저장된 field.npz 15개가 전부 위치 스크램블된 사건의 가드다.
원인은 두 개의 조용한 fallback 이었다:

  * ``_flatten_field`` 의 ``a.reshape(-1, 3)[:n_pts]``
    -> (n_pts, 3, n_pol) 버퍼를 (., 3) 으로 재해석해서 격자점 i 가 실제 점
       i // 2 의 값을 받고 격자 절반이 사라졌다.
  * ``_broadcast_pol`` 의 (n_pts, 3) -> 전 편광 복제
    -> 편광 축이 사라진 배열을 두 슬롯에 복사해 pol0 == pol1 인,
       형식상 멀쩡해 보이는 파일을 만들었다.

둘 다 이제 예외를 던져야 한다.
"""

import numpy as np
import pytest

from pymnpbem_simulation.simulation.field_calculator import FieldCalculator


N_X, N_Y, N_Z = 7, 1, 5
N_PTS = N_X * N_Y * N_Z


class GridStub(object):
    """_flatten_field / _broadcast_pol 이 쓰는 속성만 가진 최소 스텁."""

    def __init__(self, n_pts = N_PTS):
        self.grid_points = np.zeros((n_pts, 3), dtype = float)


flatten = FieldCalculator._flatten_field
broadcast = FieldCalculator._broadcast_pol


def make_field(n_pol):
    """격자점마다 고유한 값을 넣어 위치 뒤섞임을 잡아낼 수 있게 한다."""
    n = N_PTS * 3 * n_pol
    return np.arange(n, dtype = float).reshape(N_X, N_Y, N_Z, 3, n_pol)


def test_meshfield_grid_shape_maps_in_order():
    stub = GridStub()
    a = make_field(2)

    out = flatten(stub, a)

    assert out.shape == (N_PTS, 3, 2)
    # 격자점 i 는 자기 값을 받아야 한다 (i // 2 가 아니라).
    assert np.array_equal(out, a.reshape(N_PTS, 3, 2))


def test_flatten_rejects_incompatible_size():
    """격자에 안 맞는 크기는 잘라내지 말고 실패해야 한다."""
    stub = GridStub()
    a = np.zeros(((N_PTS + 3) * 3 * 2 + 1,), dtype = float)

    with pytest.raises(RuntimeError, match = 'scramble'):
        flatten(stub, a)


def test_flatten_never_produces_halved_grid():
    """예전 fallback 이 만들던 (n_pts, 3) 결과가 다시 나오면 안 된다."""
    stub = GridStub()
    a = make_field(2)

    out = flatten(stub, a)

    assert out.ndim == 3 and out.shape[2] == 2
    # 예전 버그의 결과물과 달라야 한다.
    legacy = np.asarray(a).reshape(-1, 3)[:N_PTS]
    assert not np.array_equal(out[..., 0], legacy)


def test_broadcast_pol_rejects_missing_pol_axis():
    """편광 축이 사라진 배열을 여러 편광에 복제하면 안 된다."""
    stub = GridStub()
    a = np.zeros((N_PTS, 3), dtype = complex)

    with pytest.raises(RuntimeError, match = 'polarization axis'):
        broadcast(stub, a, 2)


def test_broadcast_pol_single_pol_is_allowed():
    stub = GridStub()
    a = np.arange(N_PTS * 3, dtype = float).reshape(N_PTS, 3)

    out = broadcast(stub, a, 1)

    assert out.shape == (N_PTS, 3, 1)
    assert np.array_equal(out[..., 0], a)


def test_broadcast_pol_passes_through_matching_shape():
    stub = GridStub()
    a = np.arange(N_PTS * 3 * 2, dtype = float).reshape(N_PTS, 3, 2)

    out = broadcast(stub, a, 2)

    assert out.shape == (N_PTS, 3, 2)
    assert np.array_equal(out, a)
    # 두 편광이 서로 달라야 한다 — 복제 사고의 직접적인 가드.
    assert not np.allclose(out[..., 0], out[..., 1])
