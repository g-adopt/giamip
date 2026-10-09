"""The small NetCDF helpers that every reader and writer of the package shares.

Every NetCDF file in this package is opened through h5netcdf, a netCDF-4
layer written in Python on top of h5py. Three properties decide this:

* h5netcdf applies no mask, no scale factor and no fill value. It returns the
  numbers stored in the file. The eight GIAMIP ice files declare
  ``_FillValue = 0.0`` on ``ice_thickness``, and a reader that masks would turn
  every ice-free cell into a missing value (AGENTS.md, "Every reader in the
  package opens NetCDF through h5netcdf").
* It uses the HDF5 library that h5py brings. The Firedrake environment on Gadi
  has h5py and no netCDF4, and a second HDF5 build in the same process is a
  known source of conflicts.
* The files it writes are ordinary netCDF-4 files: netCDF-C, ncdump and xarray
  read them. tests/test_netcdf_io.py checks this with netCDF4.

Only the core API ``h5netcdf.File`` is used. The ``legacyapi`` module failed
on nested group access in h5netcdf 1.8.1.

Two details of the format are handled here, so that no reader repeats them:

* A text attribute that netCDF-C wrote comes back from h5py as ``bytes``
  (``numpy.bytes_``), and one that h5netcdf wrote comes back as ``str``.
  `attribute` turns both into ``str``.
* netCDF-C stores a numeric attribute as an array even when it holds one
  value, so a one-element array is returned as its single value.
"""

from __future__ import annotations

import pathlib

import numpy as np


def open_read(path: pathlib.Path):
    """Open a NetCDF file for reading with h5netcdf.

    Parameters
    ----------
    path
        The file.

    Returns
    -------
    h5netcdf.File
        The open file. Use it as a context manager.
    """
    import h5netcdf
    return h5netcdf.File(path, "r")


def open_write(path: pathlib.Path, append: bool):
    """Open a NetCDF file for writing with h5netcdf.

    Parameters
    ----------
    path
        The file.
    append
        True to add to an existing file, False to create a new one.

    Returns
    -------
    h5netcdf.File
        The open file. Use it as a context manager.
    """
    import h5netcdf
    return h5netcdf.File(path, "a" if append else "w")


def attribute(value):
    """One attribute value as the package uses it.

    Text becomes ``str`` whether netCDF-C (``bytes``) or h5netcdf (``str``)
    wrote it. A one-element numeric array becomes its single value, a
    NumPy scalar, which is what netCDF4's ``getncattr`` returned. Anything else
    is returned unchanged.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SO" and value.size == 1:
            item = value.reshape(-1)[0]
            return item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else item
        if value.size == 1:
            return value.reshape(-1)[0]
    return value


def attributes(obj, skip=()) -> dict:
    """The attributes of a file, group or variable as a dict, decoded.

    Parameters
    ----------
    obj
        An h5netcdf file, group or variable.
    skip
        Names to leave out.

    Returns
    -------
    dict
        Name to value, each passed through `attribute`. h5netcdf already hides
        the attributes that HDF5 and netCDF-C use for their own bookkeeping
        (``DIMENSION_LIST``, ``_Netcdf4Dimid`` and so on), except
        ``_NCProperties``, which is left out here as well: it records library
        versions and is not part of the content.
    """
    return {k: attribute(v) for k, v in obj.attrs.items()
            if k not in skip and k != "_NCProperties"}


def set_attributes(obj, values: dict) -> None:
    """Write attributes, skipping None.

    A Python ``bool`` is written as an 8-bit integer, because netCDF has no
    boolean type.
    """
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            value = np.int8(value)
        obj.attrs[key] = value


def has_group(root, group_path: str) -> bool:
    """Whether a slash-separated group path exists below an open file or group."""
    group = root
    for part in group_path.split("/"):
        if part not in group.groups:
            return False
        group = group.groups[part]
    return True


def group_at(root, group_path: str):
    """The group at a slash-separated path below an open file or group."""
    group = root
    for part in group_path.split("/"):
        group = group.groups[part]
    return group


def create_group(root, group_path: str):
    """Create the group at a slash-separated path, and any missing parents.

    The last part must not exist yet, so that a second writer never adds to a
    group another writer made (the callers check this first and raise with the
    file name).
    """
    parts = group_path.split("/")
    group = root
    for part in parts[:-1]:
        group = group.groups[part] if part in group.groups else group.create_group(part)
    return group.create_group(parts[-1])


def axis_of(group, variable) -> np.ndarray:
    """The values of the coordinate variable of a variable's first dimension.

    Every curve in this package's files is a 1-D variable whose dimension has a
    coordinate variable of the same name in the same group.

    Parameters
    ----------
    group
        The group that holds the variable.
    variable
        The variable.
    """
    return np.asarray(group.variables[variable.dimensions[0]][...])


def data_variables(group):
    """The variables of a group that are not coordinate variables, by name."""
    return {name: var for name, var in group.variables.items() if name not in group.dimensions}


def write_curves(group, curves, *, axis_name: str, coordinate_attributes: dict,
                 compress: bool = True) -> None:
    """Write 1-D curves into one group, sharing an axis wherever two are equal.

    The first distinct axis is called ``axis_name``, the next
    ``axis_name_2``, and so on, so a group whose curves are sampled alike has
    one axis. The curves are written in the order given.

    Parameters
    ----------
    group
        An open h5netcdf group.
    curves
        A sequence of ``(name, axis, values, attributes)``.
    axis_name
        The name of the first axis.
    coordinate_attributes
        The attributes of every axis variable, for example the units.
    compress
        Store the values with gzip. For curves of a few hundred points the
        uncompressed form is smaller: compression needs chunked storage, whose
        index costs more than it saves on short arrays.
    """
    axes: list[tuple[str, np.ndarray]] = []
    for name, axis, values, attrs in curves:
        axis = np.asarray(axis, dtype=float)
        dim = next((d for d, a in axes if a.shape == axis.shape and np.array_equal(a, axis)), None)
        if dim is None:
            dim = axis_name if not axes else f"{axis_name}_{len(axes) + 1}"
            axes.append((dim, axis))
            group.dimensions[dim] = axis.size
            coordinate = group.create_variable(dim, (dim,), "f8", data=axis)
            set_attributes(coordinate, coordinate_attributes)
        variable = group.create_variable(name, (dim,), "f8",
                                         data=np.asarray(values, dtype=float),
                                         compression="gzip" if compress else None)
        set_attributes(variable, attrs)
