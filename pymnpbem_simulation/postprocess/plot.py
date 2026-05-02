import os

from typing import Any, Dict

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..util import ensure_dir, print_info


def plot_spectrum(out_dir: str,
        result: Dict[str, Any],
        title: str = '') -> str:

    ensure_dir(out_dir)

    wavelength = np.asarray(result['wavelength'])
    ext = np.asarray(result['ext'])
    sca = np.asarray(result['sca'])

    n_pol = ext.shape[1]

    fig, axes = plt.subplots(1, 2, figsize = (12, 4))

    for i in range(n_pol):
        label = 'pol {}'.format(i)
        axes[0].plot(wavelength, ext[:, i], label = label)
        axes[1].plot(wavelength, sca[:, i], label = label)

    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Extinction')
    axes[0].set_title('Extinction')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Scattering')
    axes[1].set_title('Scattering')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    if title != '':
        fig.suptitle(title)

    fig.tight_layout()

    path = os.path.join(out_dir, 'spectrum.png')
    fig.savefig(path, dpi = 150)
    plt.close(fig)

    print_info('saved <{}>'.format(path))

    return path
