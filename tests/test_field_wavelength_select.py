"""Field-pass wavelength selection (``field_wavelength_idx``).

Both wrappers share one grammar for this key: ``'middle'``, a ``'peak*'``
string, a bare index, or a list of target wavelengths **in nm**. Every
config in either repo uses the nm-list or string form, so the list must
never be read as indices.
"""

import os
import sys

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pymnpbem_simulation.cli import _build_enei


def _cfg(fwi, wavelength_range = None):
    if wavelength_range is None:
        wavelength_range = [300, 1000, 21]

    return {'simulation': {
        'calculate_spectrum': False,
        'calculate_fields': True,
        'wavelength_range': wavelength_range,
        'field_wavelength_idx': fwi}}


@pytest.fixture
def spectrum_dir(tmp_path):
    wl = np.array([500.0, 550.0, 600.0])
    np.savez(os.path.join(str(tmp_path), 'spectrum.npz'),
            wavelength = wl,
            ext = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]]),
            sca = np.array([[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]]),
            abs = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]))
    return str(tmp_path)


def test_list_is_nm_not_indices():
    """A list holds wavelengths in nm, mapped to the nearest grid point."""
    # grid = linspace(300, 1000, 21) -> step 35 nm
    got = _build_enei(_cfg([593, 616]), None, None)

    assert np.allclose(got, [580.0, 615.0])


def test_list_out_of_index_range_still_resolves():
    """The regression: nm values far above len(grid) used to yield nothing."""
    got = _build_enei(_cfg(list(range(400, 1001, 30))), None, None)

    assert len(got) > 0
    assert got.min() >= 300.0 and got.max() <= 1000.0
    # duplicates collapse: 21 targets onto a 21-point grid, step 35 vs 30
    assert len(got) == len(np.unique(got))


def test_middle_picks_grid_centre():
    got = _build_enei(_cfg('middle'), None, None)

    assert np.allclose(got, [650.0])


def test_bare_int_is_an_index():
    got = _build_enei(_cfg(3), None, None)

    assert np.allclose(got, [405.0])


@pytest.mark.parametrize('key,expected', [
    ('peak', [600.0]),
    ('peak_abs', [600.0]),
    ('peak_ext', [500.0, 600.0]),
    ('peak_sca', [550.0])])
def test_peak_forms_read_the_spectrum(spectrum_dir, key, expected):
    got = _build_enei(_cfg(key, [500, 600, 3]), None, spectrum_dir)

    assert np.allclose(got, expected)


def test_peak_without_spectrum_raises():
    with pytest.raises(ValueError, match = 'spectrum'):
        _build_enei(_cfg('peak'), None, None)


def test_unknown_string_raises():
    with pytest.raises(ValueError, match = 'field_wavelength_idx'):
        _build_enei(_cfg('bogus'), None, None)


def test_out_of_range_index_raises():
    with pytest.raises(ValueError, match = 'out of range'):
        _build_enei(_cfg(999), None, None)


def test_explicit_field_wavelengths_wins():
    cfg = _cfg([593, 616])
    cfg['simulation']['field_wavelengths'] = [777.0]

    assert np.allclose(_build_enei(cfg, None, None), [777.0])


def test_spectrum_mode_ignores_field_keys():
    cfg = _cfg([593, 616])
    cfg['simulation']['calculate_spectrum'] = True

    assert len(_build_enei(cfg, None, None)) == 21
