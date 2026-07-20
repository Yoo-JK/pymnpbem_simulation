import os
import sys
import json
import time
import random

from typing import Any, Dict, List, Tuple, Optional, Callable

import numpy as np


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok = True)


def now_str() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())


def save_json(path: str,
        obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    for attempt in range(3):
        try:
            with open(path, 'w') as f:
                json.dump(obj, f, indent = 2, default = _json_default)
            return
        except OSError:
            print('[error] save_json failed, retry {}/3'.format(attempt + 1))
            time.sleep(10)
    raise OSError('[error] save_json: failed to write <{}>'.format(path))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    raise TypeError('[error] unsupported JSON type: <{}>'.format(type(obj).__name__))


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def grade_diff(rel: float) -> str:
    if rel < 1e-12:
        return 'machine'
    if rel < 1e-9:
        return 'OK'
    if rel < 1e-6:
        return 'good'
    if rel < 1e-3:
        return 'warn'
    return 'BAD'


def print_info(msg: str) -> None:
    print('[info] {}'.format(msg), flush = True)


def print_error(msg: str) -> None:
    print('[error] {}'.format(msg), flush = True)

def assert_no_callables(obj: Any,
        path: str = 'cfg') -> None:
    """Fail fast on raw callables in config structures sent to workers.

    Multiprocessing with spawn pickles target args; direct callable objects in
    cfg can crash with opaque pickling errors. Users should provide descriptor
    dicts (e.g. type=python_module) so each worker resolves callables locally.
    """
    if callable(obj):
        raise TypeError(
                '[error] Unpicklable callable found at <{}> in config. '
                'For multi-worker/multi-node runs, use '
                '<materials.refractive_index_paths> descriptors '
                '(type=constant/table/python_module) instead of embedding '
                'raw callables in cfg.'.format(path))

    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_callables(v, '{}.{}'.format(path, k))
        return

    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            assert_no_callables(v, '{}[{}]'.format(path, i))
