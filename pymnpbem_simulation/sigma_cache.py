"""
Sigma cache — store BEM solver state (sig1, sig2, h1, h2 for retarded;
sig for quasistatic) per (wavelength, polarization, propagation_dir)
to disk so downstream analyses (field evaluation, surface charge map,
EELS) can be re-run without repeating the BEM solve.

Folder layout (rooted at <output_dir>/sigma/):

  sigma/
      manifest.json
      wl_{nm:07.2f}_p{pxyz}_d{dxyz}.npz   # one file per (wl, pol, dir)
      ...

File naming:
  - wavelength: zero-padded to 2 decimals, e.g. wl_0553.00 (sorts lexically)
  - polarization / propagation: 3-digit signed encoding (see _encode_vec3)
      [1, 0, 0]   -> p100 / d100
      [0, -1, 0]  -> pn10 / dn10   ('n' prefix per negative component is awkward;
                                    we use signed integer codes instead — see below)
  - actual encoding used: each component mapped to one of {'p','z','n'} for
    +/0/-, then concatenated.  Example: pol [1, 0, 0] -> 'pzz'.  This keeps
    filenames short and unambiguous.

Inside each npz:
  retarded solver:
    sig1: complex128, shape (n_faces,)
    sig2: complex128, shape (n_faces,)
    h1:   complex128, shape (n_faces, 3)
    h2:   complex128, shape (n_faces, 3)
    wavelength_nm: float
    pol: int8, shape (3,)        # original polarization vector (-1/0/+1)
    prop_dir: int8, shape (3,)   # original propagation direction
    solver_type: str (b'retarded')

  quasistatic solver:
    sig:  complex128, shape (n_faces,)
    wavelength_nm, pol, prop_dir, solver_type ('quasistatic')

Manifest (manifest.json) — metadata only, regenerated from on-disk files:
  {
    "version": "1.0",
    "n_faces": <int>,
    "solver_type": "retarded" | "quasistatic",
    "structure_hash": "<sha256>",
    "eps_hash": "<sha256>",
    "excitations": [{"pol": [...], "prop_dir": [...]}, ...],
    "wavelengths_nm": [...],
    "last_updated": "<isoformat>"
  }
"""

import os
import json
import glob
import hashlib
import datetime
import time

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


SIGMA_VERSION = '1.0'
_MANIFEST_LOCK_TIMEOUT_S = 10.0
_MANIFEST_LOCK_POLL_S = 0.05


def _encode_axis_component(v: float) -> str:
    """Map a single axis component to one char.

    +1/positive -> '1', 0 -> '0', -1/negative -> 'n'.  Components are
    expected to be -1, 0, or +1 (axis-aligned polarization/dir).  The
    'n' prefix marks a negative axis component, keeping filenames
    unambiguous without using '-' (which can confuse shell tokenizers).
    """

    if v > 0.5:
        return '1'
    elif v < -0.5:
        return 'n'
    else:
        return '0'


def _encode_vec3(vec: Sequence[float]) -> str:
    """Encode a 3-vector to a 3-char string (e.g. [1,0,0] -> '100')."""

    assert len(vec) == 3, '[error] vec must have length 3'
    return ''.join(_encode_axis_component(float(c)) for c in vec)


def _sanitize_int_vec(vec: Sequence[float]) -> List[int]:
    """Coerce a 3-vector to signed integers (-1/0/+1)."""

    out = []
    for c in vec:
        c = float(c)
        if c > 0.5:
            out.append(1)
        elif c < -0.5:
            out.append(-1)
        else:
            out.append(0)
    return out


def make_filename(wavelength_nm: float,
        pol: Sequence[float],
        prop_dir: Sequence[float]) -> str:
    """Build the sigma file basename for one (wl, pol, dir) tuple."""

    return 'wl_{:07.2f}_p{}_d{}.npz'.format(
            float(wavelength_nm),
            _encode_vec3(pol),
            _encode_vec3(prop_dir))


def sigma_dir(output_dir: str) -> str:
    """Return the sigma subfolder path (does NOT create)."""

    return os.path.join(output_dir, 'sigma')


def ensure_sigma_dir(output_dir: str) -> str:
    """Create and return the sigma subfolder path."""

    d = sigma_dir(output_dir)
    os.makedirs(d, exist_ok = True)
    return d


def _decode_axis_component(token: str) -> int:
    """Decode one filename axis token back to -1/0/+1."""

    if token == '1':
        return 1
    if token == 'n':
        return -1
    if token == '0':
        return 0
    raise ValueError('[error] Unknown sigma axis token <{}>'.format(token))


def _decode_vec3(token: str) -> List[int]:
    """Decode a 3-char filename vector token."""

    if len(token) != 3:
        raise ValueError('[error] Sigma vec token must be 3 chars, got <{}>'.format(token))
    return [_decode_axis_component(ch) for ch in token]


# ---------------------------------------------------------------------------
# Hash helpers — used by manifest for invalidation.
# ---------------------------------------------------------------------------

def _yaml_section_hash(section: Any) -> str:
    """Stable SHA256 of a yaml-compatible structure.

    Uses safe_dump with sorted keys for determinism.
    """

    if section is None:
        return ''
    try:
        text = yaml.safe_dump(section, sort_keys = True, default_flow_style = False)
    except Exception:
        text = str(section)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def compute_structure_hash(cfg_struct: Dict[str, Any]) -> str:
    return _yaml_section_hash(cfg_struct)


def compute_eps_hash(cfg_materials: Dict[str, Any]) -> str:
    return _yaml_section_hash(cfg_materials)


# ---------------------------------------------------------------------------
# Per-polarization save: split sig (multi-RHS) into per-pol files.
# ---------------------------------------------------------------------------

def save_sigma_per_pol(output_dir: str,
        sig: Any,
        wavelength_nm: float,
        polarizations: Sequence[Sequence[float]],
        propagation_dirs: Sequence[Sequence[float]],
        solver_type: str = 'retarded') -> List[str]:
    """Save one wavelength's sig, splitting along the polarization axis.

    For retarded solver: sig has fields sig1, sig2, h1, h2.  Shapes:
      sig1, sig2: (n_faces, npol)
      h1, h2:    (n_faces, 3, npol)

    We slice the trailing pol axis to get per-pol arrays of shape
    (n_faces,) and (n_faces, 3), then write one npz per pol.

    Skips files that already exist (idempotent — caller controls re-save
    via deleting the file or passing force=True via overwrite=True kwarg).

    Returns the list of file paths written (or already-existing).
    """

    d = ensure_sigma_dir(output_dir)
    npol = len(polarizations)
    assert npol == len(propagation_dirs), \
            '[error] polarizations / propagation_dirs length mismatch'

    paths = []

    if solver_type == 'retarded':
        for k in range(npol):
            pol_k = _sanitize_int_vec(polarizations[k])
            dir_k = _sanitize_int_vec(propagation_dirs[k])
            fname = make_filename(wavelength_nm, pol_k, dir_k)
            fpath = os.path.join(d, fname)
            paths.append(fpath)
            if os.path.exists(fpath):
                continue

            # Slice the multi-RHS sig along its trailing pol axis.
            # If sig was solved single-pol, shapes are (n_faces,) and
            # (n_faces, 3) directly.
            sig1_arr = np.asarray(sig.sig1)
            sig2_arr = np.asarray(sig.sig2)
            h1_arr = np.asarray(sig.h1)
            h2_arr = np.asarray(sig.h2)

            if sig1_arr.ndim == 2:
                sig1_k = sig1_arr[:, k]
                sig2_k = sig2_arr[:, k]
            else:
                sig1_k = sig1_arr
                sig2_k = sig2_arr

            if h1_arr.ndim == 3:
                h1_k = h1_arr[:, :, k]
                h2_k = h2_arr[:, :, k]
            else:
                h1_k = h1_arr
                h2_k = h2_arr

            np.savez(fpath,
                    sig1 = sig1_k.astype(np.complex128),
                    sig2 = sig2_k.astype(np.complex128),
                    h1 = h1_k.astype(np.complex128),
                    h2 = h2_k.astype(np.complex128),
                    wavelength_nm = float(wavelength_nm),
                    pol = np.array(pol_k, dtype = np.int8),
                    prop_dir = np.array(dir_k, dtype = np.int8),
                    solver_type = 'retarded')

    elif solver_type == 'quasistatic':
        for k in range(npol):
            pol_k = _sanitize_int_vec(polarizations[k])
            dir_k = _sanitize_int_vec(propagation_dirs[k])
            fname = make_filename(wavelength_nm, pol_k, dir_k)
            fpath = os.path.join(d, fname)
            paths.append(fpath)
            if os.path.exists(fpath):
                continue

            sig_arr = np.asarray(sig.sig)
            if sig_arr.ndim == 2:
                sig_k = sig_arr[:, k]
            else:
                sig_k = sig_arr

            np.savez(fpath,
                    sig = sig_k.astype(np.complex128),
                    wavelength_nm = float(wavelength_nm),
                    pol = np.array(pol_k, dtype = np.int8),
                    prop_dir = np.array(dir_k, dtype = np.int8),
                    solver_type = 'quasistatic')

    else:
        raise ValueError(
                '[error] Unknown solver_type for sigma save: <{}>'.format(solver_type))

    return paths


# ---------------------------------------------------------------------------
# Load: reconstruct multi-pol sig from per-pol files (for field-only mode).
# ---------------------------------------------------------------------------

def load_sigma(output_dir: str,
        wavelength_nm: float,
        polarizations: Sequence[Sequence[float]],
        propagation_dirs: Sequence[Sequence[float]]) -> Optional[Dict[str, np.ndarray]]:
    """Load sigma for one wavelength across requested polarizations.

    Returns a dict with stacked arrays:
      retarded: {sig1, sig2, h1, h2, solver_type='retarded', n_faces}
      quasistatic: {sig, solver_type='quasistatic', n_faces}

    Returns None if any required file is missing.
    """

    d = sigma_dir(output_dir)
    if not os.path.isdir(d):
        return None

    npol = len(polarizations)
    arrs: Dict[str, List[np.ndarray]] = {}
    solver_type = None

    for k in range(npol):
        pol_k = _sanitize_int_vec(polarizations[k])
        dir_k = _sanitize_int_vec(propagation_dirs[k])
        fname = make_filename(wavelength_nm, pol_k, dir_k)
        fpath = os.path.join(d, fname)

        if not os.path.exists(fpath):
            return None

        data = np.load(fpath)
        st = str(data['solver_type'].item() if hasattr(data['solver_type'], 'item')
                else data['solver_type'])
        if solver_type is None:
            solver_type = st
        elif solver_type != st:
            raise ValueError(
                    '[error] Mixed solver_type in sigma cache (got <{}> and <{}>)'.format(
                            solver_type, st))

        if solver_type == 'retarded':
            for key in ('sig1', 'sig2', 'h1', 'h2'):
                arrs.setdefault(key, []).append(np.array(data[key]))
        else:
            arrs.setdefault('sig', []).append(np.array(data['sig']))

    out: Dict[str, np.ndarray] = {'solver_type': solver_type}
    if solver_type == 'retarded':
        out['sig1'] = np.stack(arrs['sig1'], axis = -1)
        out['sig2'] = np.stack(arrs['sig2'], axis = -1)
        out['h1'] = np.stack(arrs['h1'], axis = -1)
        out['h2'] = np.stack(arrs['h2'], axis = -1)
        out['n_faces'] = int(out['sig1'].shape[0])
    else:
        out['sig'] = np.stack(arrs['sig'], axis = -1)
        out['n_faces'] = int(out['sig'].shape[0])

    return out


# ---------------------------------------------------------------------------
# Manifest helpers.
# ---------------------------------------------------------------------------

def manifest_path(output_dir: str) -> str:
    return os.path.join(sigma_dir(output_dir), 'manifest.json')


def _manifest_lock_path(output_dir: str) -> str:
    return os.path.join(sigma_dir(output_dir), 'manifest.lock')


@contextmanager
def _manifest_lock(output_dir: str,
        timeout_s: float = _MANIFEST_LOCK_TIMEOUT_S,
        poll_s: float = _MANIFEST_LOCK_POLL_S):
    """Serialize manifest updates across workers with a lock file."""

    ensure_sigma_dir(output_dir)
    lock_path = _manifest_lock_path(output_dir)
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    fd = None

    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode('ascii', errors = 'ignore'))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                        '[error] Timed out waiting for sigma manifest lock <{}>'.format(
                                lock_path))
            time.sleep(max(0.001, float(poll_s)))

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _write_manifest_payload(path: str, payload: Dict[str, Any]) -> None:
    """Atomically replace manifest.json so readers never see partial JSON."""

    d = os.path.dirname(path)
    os.makedirs(d, exist_ok = True)
    tmp_path = os.path.join(d, 'manifest.{}.tmp'.format(os.getpid()))
    try:
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent = 2, sort_keys = False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _scan_cached_manifest_entries(output_dir: str) -> Tuple[List[Dict[str, List[int]]], List[float]]:
    """Recover manifest metadata from cached per-wavelength sigma files."""

    d = sigma_dir(output_dir)
    excitations: List[Dict[str, List[int]]] = []
    wavelengths_nm = set()

    for fpath in sorted(glob.glob(os.path.join(d, 'wl_*.npz'))):
        base = os.path.basename(fpath)
        stem, ext = os.path.splitext(base)
        if ext.lower() != '.npz':
            continue

        parts = stem.split('_')
        if len(parts) != 4:
            continue

        try:
            wl_nm = round(float(parts[1]), 4)
            pol = _decode_vec3(parts[2][1:])
            prop_dir = _decode_vec3(parts[3][1:])
        except (IndexError, ValueError):
            continue

        item = {'pol': pol, 'prop_dir': prop_dir}
        if item not in excitations:
            excitations.append(item)
        wavelengths_nm.add(wl_nm)

    return excitations, sorted(wavelengths_nm)


def _recover_manifest_from_disk(output_dir: str,
        n_faces: int,
        solver_type: str,
        structure_hash: str,
        eps_hash: str) -> Dict[str, Any]:
    """Rebuild manifest metadata from on-disk sigma files after corruption."""

    excitations, wavelengths_nm = _scan_cached_manifest_entries(output_dir)
    write_manifest(
            output_dir,
            n_faces = n_faces,
            solver_type = solver_type,
            structure_hash = structure_hash,
            eps_hash = eps_hash,
            excitations = excitations,
            wavelengths_nm = wavelengths_nm)
    return read_manifest(output_dir) or {
            'version': SIGMA_VERSION,
            'n_faces': int(n_faces),
            'solver_type': str(solver_type),
            'structure_hash': str(structure_hash),
            'eps_hash': str(eps_hash),
            'excitations': excitations,
            'wavelengths_nm': wavelengths_nm}


def write_manifest(output_dir: str,
        n_faces: int,
        solver_type: str,
        structure_hash: str,
        eps_hash: str,
        excitations: List[Dict[str, List[int]]],
        wavelengths_nm: List[float]) -> str:
    d = ensure_sigma_dir(output_dir)
    payload = {
            'version': SIGMA_VERSION,
            'n_faces': int(n_faces),
            'solver_type': str(solver_type),
            'structure_hash': str(structure_hash),
            'eps_hash': str(eps_hash),
            'excitations': excitations,
            'wavelengths_nm': sorted({round(float(w), 4) for w in wavelengths_nm}),
            'last_updated': datetime.datetime.now().isoformat(timespec = 'seconds')}
    path = manifest_path(output_dir)
    _write_manifest_payload(path, payload)
    return path


def read_manifest(output_dir: str) -> Optional[Dict[str, Any]]:
    path = manifest_path(output_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def update_manifest_append(output_dir: str,
        n_faces: int,
        solver_type: str,
        structure_hash: str,
        eps_hash: str,
        polarizations: Sequence[Sequence[float]],
        propagation_dirs: Sequence[Sequence[float]],
        wavelength_nm: float) -> str:
    """Append one wavelength (and any new pol/dir) to the manifest.

    Called after each per-wl sigma dump so a crash mid-sweep leaves the
    manifest in a consistent state.
    """

    with _manifest_lock(output_dir):
        try:
            existing = read_manifest(output_dir)
        except json.JSONDecodeError:
            existing = _recover_manifest_from_disk(
                    output_dir,
                    n_faces = n_faces,
                    solver_type = solver_type,
                    structure_hash = structure_hash,
                    eps_hash = eps_hash)

        if existing is None:
            existing = {
                    'version': SIGMA_VERSION,
                    'n_faces': int(n_faces),
                    'solver_type': str(solver_type),
                    'structure_hash': str(structure_hash),
                    'eps_hash': str(eps_hash),
                    'excitations': [],
                    'wavelengths_nm': []}

        # Hash mismatch: refuse to corrupt — bail out, caller decides recovery.
        if existing.get('structure_hash') and existing['structure_hash'] != structure_hash:
            raise ValueError(
                    '[error] sigma manifest structure_hash mismatch — refuse to append. '
                    'Existing dir <{}>'.format(sigma_dir(output_dir)))
        if existing.get('eps_hash') and existing['eps_hash'] != eps_hash:
            raise ValueError(
                    '[error] sigma manifest eps_hash mismatch — refuse to append. '
                    'Existing dir <{}>'.format(sigma_dir(output_dir)))

        excs = list(existing.get('excitations', []))
        for pol, prop in zip(polarizations, propagation_dirs):
            item = {
                    'pol': _sanitize_int_vec(pol),
                    'prop_dir': _sanitize_int_vec(prop)}
            if item not in excs:
                excs.append(item)

        wls = set(round(float(w), 4) for w in existing.get('wavelengths_nm', []))
        wls.add(round(float(wavelength_nm), 4))

        return write_manifest(
                output_dir,
                n_faces = n_faces,
                solver_type = solver_type,
                structure_hash = structure_hash,
                eps_hash = eps_hash,
                excitations = excs,
                wavelengths_nm = sorted(wls))


# ---------------------------------------------------------------------------
# Discovery: list cached wavelengths on disk for a given pol/dir set.
# ---------------------------------------------------------------------------

def find_cached_wavelengths(output_dir: str,
        polarizations: Sequence[Sequence[float]],
        propagation_dirs: Sequence[Sequence[float]]) -> List[float]:
    """Return wavelengths for which ALL requested pol/dir files exist."""

    d = sigma_dir(output_dir)
    if not os.path.isdir(d):
        return []

    # Glob files for the first (pol, dir) — they constrain wavelength set.
    pol0 = _sanitize_int_vec(polarizations[0])
    dir0 = _sanitize_int_vec(propagation_dirs[0])
    pat = os.path.join(d, 'wl_*_p{}_d{}.npz'.format(
            _encode_vec3(pol0), _encode_vec3(dir0)))

    wls: List[float] = []
    for fpath in sorted(glob.glob(pat)):
        base = os.path.basename(fpath)
        try:
            wl_str = base.split('_')[1]
            wl_val = float(wl_str)
        except (IndexError, ValueError):
            continue

        # Verify all other (pol, dir) exist for this wavelength.
        ok = True
        for k in range(1, len(polarizations)):
            polk = _sanitize_int_vec(polarizations[k])
            dirk = _sanitize_int_vec(propagation_dirs[k])
            other = make_filename(wl_val, polk, dirk)
            if not os.path.exists(os.path.join(d, other)):
                ok = False
                break

        if ok:
            wls.append(wl_val)

    return wls
