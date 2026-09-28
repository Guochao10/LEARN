"""MAT-file I/O shared by the original and PyTorch implementations."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat


def _decode_hdf5_complex(value: np.ndarray) -> np.ndarray:
    if value.dtype.fields and {"real", "imag"}.issubset(value.dtype.fields):
        return value["real"] + 1j * value["imag"]
    return value


def load_mat_array(path: str | Path, key: str) -> np.ndarray:
    """Read an array from MATLAB v5/v7 or v7.3 files.

    MATLAB v7.3 stores dimensions in reverse order in its HDF5 datasets, so
    the axes are reversed when that format is detected.
    """
    path = Path(path)
    try:
        contents = loadmat(path)
        if key not in contents:
            raise KeyError(f"{key!r} not found in {path}")
        return np.asarray(contents[key])
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"{key!r} not found in {path}")
        node = handle[key]
        if isinstance(node, h5py.Group):
            if not {"real", "imag"}.issubset(node.keys()):
                raise TypeError(f"unsupported HDF5 group layout for {key!r} in {path}")
            value = np.asarray(node["real"]) + 1j * np.asarray(node["imag"])
        else:
            value = _decode_hdf5_complex(np.asarray(node))
    return np.asarray(value).transpose(tuple(range(value.ndim - 1, -1, -1)))


def save_complex_mat(path: str | Path, key: str, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {key: np.asarray(value)}, do_compression=True)

