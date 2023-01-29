# Copyright (c) 2023 Taner Esat <t.esat@fz-juelich.de>

import numpy as np


def getSpecgridData(filename):
    """Loads a Createc spectroscopy grid file (*.specgrid).

    Args:
        filename (str): Path and filename.

    Returns:
        ndarray, ndarray, ndarray: Bias voltage, Z-values, Specgrid data as an array with format [y, x, channel, data array]
    """    
    header_size = 1024

    # Header
    header = np.fromfile(filename, dtype=np.int32, count=header_size)
    version = header[0]
    nx = header[1]
    ny = header[2]
    specxgrid = header[5]
    specygrid = header[6]
    vertpoints = header[7]
    specgridchan = header[14]
    # specgridchannelselectva = header[14]
    if version == 4:
        xend = header[19]
        yend = header[21]
    else:
        xend = int(np.round(nx/specxgrid))
        yend = int(np.round(ny/specygrid))

    # V and Z Tables
    v_z_table_raw = np.fromfile(filename, dtype=np.float32, count=2*vertpoints, offset=header_size)
    spectra_raw = np.fromfile(filename, dtype=np.float32, count=-1, offset=header_size+4*2*vertpoints)
    
    v_table = v_z_table_raw[0:2*vertpoints:2]
    z_table = v_z_table_raw[1:2*vertpoints:2]

    # All spectra in the grid
    specgrid = np.zeros(shape=(yend, xend, specgridchan, vertpoints))
    for y in range(0, yend):
        for x in range(0, xend):
                for channel in range(0, specgridchan):
                    size_single_spec = specgridchan*vertpoints
                    start = y * xend * size_single_spec + x * size_single_spec
                    end = start + size_single_spec
                    specgrid[y, x, channel] = spectra_raw[start+channel:end:specgridchan]
    
    return v_table, z_table, specgrid