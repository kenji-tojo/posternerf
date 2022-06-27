import numpy as np
import math
from typing import Tuple

from ._bin import plenoctree as _libplnoct


def generate_dirs_cylindrical(res_theta: int, hemisphere: bool = True) -> np.ndarray:
    print(f'generating {res_theta}x{4*res_theta}x4 directions for anti-aliased radiance sampling')

    n = res_theta * 2
    u = (np.arange(n) + 0.5) / n
    v = (np.arange(n*4) + 0.5) / (n*4)

    theta = np.arccos(u) if hemisphere else np.arccos(u*2-1)
    phi = 2*math.pi*v

    dirs = []
    for i_t in range(res_theta):
        for i_p in range(res_theta*4):
            for t in theta[i_t*2:i_t*2+2]:
                for p in phi[i_p*2:i_p*2+2]:
                    dirs.append(np.array([
                        math.sin(t) * math.cos(p),
                        math.sin(t) * math.sin(p),
                        math.cos(t)]))
    dirs = np.vstack(dirs)

    return dirs.reshape(4*res_theta**2, 4, 3)

def compute_RGB_histogram(
    colors_rgb: np.ndarray,
    weights: np.ndarray,
    bits_per_channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3
    assert weights.ndim == 1
    assert len(colors_rgb) == len(weights)
    assert 1 <= bits_per_channel and bits_per_channel <=8

    try:
        bin_weights, bin_centers_rgb = _libplnoct.compute_RGB_histogram(
            colors_rgb.flatten(), weights.flatten(), bits_per_channel)
    except RuntimeError as err:
        print(err)
        assert False

    return bin_weights, bin_centers_rgb
