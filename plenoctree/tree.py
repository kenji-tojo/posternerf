import numpy as np
from ._bin import plenoctree as _libplnoct

from typing import Tuple


def _print_large_array(name: str, array: np.ndarray):
    print(f'  {name}: ndarray of shape {array.shape} and dtype {array.dtype}')

def _load_npz(path: str):
    files = np.load(path)
    print(f'files in {path}')
    for f in files:
        if len(files[f].shape) > 0:
            _print_large_array(f, files[f])
        else:
            print(f'  {f}: {files[f]}')
    return files

def _parse_format(format: str) -> int:
    assert format == 'SH9'
    return 9

class PlenOctree():
    def __init__(self, path: str) -> None:
        print(f'loading the plenoctree model from {path}')
        files = _load_npz(path)

        self.path: str         = path
        self.data: np.ndarray  = files['data']
        self.data_dim: int     = files['data_dim']
        self.data_format: str  = files['data_format']
        self.basis_dim: int    = _parse_format(self.data_format)
        self.child: np.ndarray = files['child']
        self.shift: np.ndarray = files['offset']
        self.scale: np.ndarray = files['invradius3']

        ## checking assumptions our cpp code makes
        assert self.child.size * self.data_dim == self.data.size
        assert self.shift.size == 3 and self.scale.size == 3

        self.print()
    
    def print(self):
        print('PlenOctree instance')
        print(f'  source path: {self.path}')
        _print_large_array('data', self.data)
        print(f'  data_dim: {self.data_dim}')
        print(f'  basis_dim: {self.basis_dim}')
        _print_large_array('child', self.child)
        print(f'  shift: {self.shift}')
        print(f'  scale: {self.scale}')
    
    def show(
        self,
        palette: np.ndarray = np.array([], dtype=np.float32),
        output_dir: str = ''
    ):
        try:
            _libplnoct.launch_viewer(
                self.data_dim, self.basis_dim,
                self.data.flatten(), self.child.flatten(),
                self.shift.flatten(), self.scale.flatten(),
                palette.flatten(),
                output_dir)
        except RuntimeError as err:
            print(err)
            assert False
    
    def sample_radiance(self, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            num_dirs_final = len(dirs)
            colors_raw, colors_aa, weights = _libplnoct.sample_radiance(
                self.data_dim, self.basis_dim,
                self.data.flatten(), self.child.flatten(),
                self.shift.flatten(), self.scale.flatten(),
                num_dirs_final,
                dirs.flatten())
        except RuntimeError as err:
            print(err)
            assert False
        
        return colors_raw, colors_aa, weights


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_path', help='path to the plenoctree npz file')
    args = parser.parse_args()

    tree = PlenOctree(args.npz_path)
    tree.show()
