# Copyright (c) 2023 Taner Esat <t.esat@fz-juelich.de>

import os

import numpy as np

def getFileHeader(filename, endTag):
    """Loads the header information from a Nanonis file.

    Args:
        filename (str): Path and filename.
        endTag (str): Tag describing end of header.

    Returns:
        dict: Header information.
    """    
    header_lines = 0
    header_size = 0
    header = {}
    if os.path.exists(filename):
        with open(filename, errors="replace") as f:
            for line in f:
                header_size = len(line.encode()) + header_size + 1
                header_lines += 1
                if endTag in line:
                    break
                if len(line) > 2:
                    line = line.replace('\n', '')
                    line = line.replace('\r', '')
                    line = line.replace('"', '')
                    param, val = line.split('=', 1)
                    header[param] = val
            
            header['Header size in bytes'] = header_size
            header['Header number of lines'] = header_lines
    
    return header

def getSpectroscopyData(filename):
    """Loads a Nanonis spectroscopy (*.dat) file.

    Args:
        filename (str): Path and filename.

    Returns:
        dict, ndarray: Header information, Spectroscopy data.
    """    
    header = {}
    data = []
    if os.path.exists(filename):
        header = getFileHeader(filename, '[DATA]')        
        data = np.genfromtxt(filename, skip_header=header['Header number of lines']+1, delimiter='\t')

    return header, data

def getGridData(filename):
    """Loads a Nanonis Grid File (*.3ds).

    Args:
        filename (str): Path and filename.

    Returns:
        dict, dict, dict, dict: Header information, Sweep signal, Experimental paramaters, Grid data.
    """    
    header = {}
    sweepSignal = {}
    expParams = {}
    gridData = {}
    if os.path.exists(filename):
        header = getFileHeader(filename, ':HEADER_END:')   
    
        numParams = int(header['# Parameters (4 byte)'])
        numPoints = int(header['Points'])
        nx = int(header['Grid dim'].split(' x ')[0])
        ny = int(header['Grid dim'].split(' x ')[1])
        channels = header['Channels'].split(';')
        numChannels = len(channels)
        params = header['Fixed parameters'].split(';') + header['Experiment parameters'].split(';')

        rawData = np.fromfile(filename, dtype='>f4', offset=header['Header size in bytes'])
        rawData.resize((ny, nx, numParams + numChannels*numPoints))

        for i, par in enumerate(params):
            expParams[par] = rawData[:, :, i]

        sweepSignal[header['Sweep Signal']] = np.linspace(expParams['Sweep Start'][0,0], expParams['Sweep End'][0,0], numPoints)

        for i, chann in enumerate(channels):
            start_ind = numParams + i * numPoints
            stop_ind = numParams + (i+1) * numPoints
            gridData[chann] = rawData[:, :, start_ind:stop_ind]
       
    return header, sweepSignal, expParams, gridData