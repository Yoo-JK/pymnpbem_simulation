from typing import Any, Dict

import numpy as np

from ..util import print_info


def dispatch_single_node(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    n_workers = int(cfg['compute']['n_workers'])
    n_gpus_per_worker = int(cfg['compute']['n_gpus_per_worker'])

    print_info('dispatch_single_node: n_workers={}, n_gpus_per_worker={}, n_wl={}'.format(
        n_workers, n_gpus_per_worker, len(enei)))

    if n_workers == 1 and n_gpus_per_worker == 0:
        return _serial_cpu(cfg, p, epstab, enei)

    if n_workers >= 1 and n_gpus_per_worker == 1:
        print_info(
            '[warn] multi-GPU dispatch not yet wired in Wave 1; falling back to serial CPU')
        return _serial_cpu(cfg, p, epstab, enei)

    if n_workers > 1 and n_gpus_per_worker == 0:
        print_info(
            '[warn] CPU process-pool dispatch not yet wired in Wave 1; falling back to serial')
        return _serial_cpu(cfg, p, epstab, enei)

    raise NotImplementedError(
        '[error] Unsupported dispatch combo (n_workers={}, n_gpus_per_worker={})'.format(
            n_workers, n_gpus_per_worker))


def _serial_cpu(cfg: Dict[str, Any],
        p: Any,
        epstab: Any,
        enei: np.ndarray) -> Dict[str, Any]:

    from ..simulation.base import run_simulation

    return run_simulation(cfg, p, epstab, enei)
