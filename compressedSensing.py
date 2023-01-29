# Copyright (c) 2023 Taner Esat <t.esat@fz-juelich.de>

import json

import matplotlib.pyplot as plt
import numpy as np
import pylops
import scipy.fft as ft
import scipy.fftpack as spfft
from spgl1 import spg_bpdn


def reconstruct2D(nx, ny, b, sampleIndices, sigma=0.01, iterations=500):
    """Reconstructs the "full" 2D data from the sparse sampled dataset.

    Args:
        nx (int): Number of points in x direction.
        ny (int): Number of points in y direction.
        b (ndarray): Sampled data as flattened array. For better performance, the data should be normalized between 0 and 1.
        sampleIndices (ndarray): Indices of sampled points.
        sigma (float, optional): Condition for minimization ||A x - b|| <= sigma. Defaults to 0.01.
        iterations (int, optional): Number of iterations for solver. Defaults to 500.

    Returns:
        ndarray, ndarray: Reconstructed 2D-Array, Fourier transform of reconstructed 2D-Array.
    """    
    xDomainTransform = spfft.idct(np.identity(nx), norm='ortho', axis=0).astype(np.float32)
    if nx == ny:
        yDomainTransform = xDomainTransform
    else:
        yDomainTransform = spfft.idct(np.identity(ny), norm='ortho', axis=0).astype(np.float32)

    idctOpX = pylops.MatrixMult(xDomainTransform)
    idctOpY = pylops.MatrixMult(yDomainTransform)
    PsiOp = pylops.Kronecker(idctOpY, idctOpX)
    PhiOp = pylops.Restriction(nx*ny, sampleIndices)
    A = PhiOp * PsiOp

    x, resid, grad, info = spg_bpdn(A, b, sigma, iter_lim=iterations,verbosity=2)

    Xdct = np.array(x).reshape(ny, nx)
    Xrec = ft.idctn(Xdct, norm="ortho")

    FTXrec = np.fft.fft2(Xrec-np.mean(Xrec))
    FTXrec = np.fft.fftshift(FTXrec)

    return Xrec, FTXrec

def reconstructGrid(grid, nx, ny, points, sampleIndices, sigma=0.01, iterations=500):
    """Reconstructs the "complete" 3D data from the sparsely sampled grid. The reconstruction is done in 2D, slice by slice. The data is sparse in the x and -y directions. The z-direction (third dimension) is complete.

    Args:
        grid (ndarray): Grid data in "flattened" form: [1, len(sampleIndices), points]
        nx (int): Number of points in x direction.
        ny (int): Number of points in y direction.
        points (int): Number of slices in (z direction) third dimension.
        sampleIndices (ndarray): Indices of sampled points.
        sigma (float, optional): Condition for minimization ||A x - b|| <= sigma. Defaults to 0.01.
        iterations (int, optional): Number of iterations for solver. Defaults to 500.

    Returns:
        ndarray: Reconstructed grid data.
    """    
    gridData = (grid - np.min(grid))/np.max(grid)

    gridRec = np.zeros(shape=(ny, nx, points))
    for i in range(points):
        b = gridData[0,:,i]
        Xrec, FTXrec = reconstruct2D(nx, ny, b, sampleIndices, iterations=100, sigma=1e-2)
        gridRec[:,:,i] = Xrec

    return gridRec

def generateRandomUniformSampleIndices(nx, ny, p):
    """Generate random uniform indices for sampling.

    Args:
        nx (int): Number of points in x direction.
        ny (int): Number of points in y direction.
        p (float): Percentage to be sampled.

    Returns:
        ndarray: Random indices (1D).
    """    
    n = round(nx * ny * p)
    idx = np.random.choice(nx * ny, n, replace=False)
    idx = np.sort(idx) 
    return idx

def sample2D(X, idx):
    """Extract/sample random indices (1D) from X. For sampling X is flattened beforehand.

    Args:
        X (ndarray): 2D-Array.
        idx (ndarray): Random indices (1D) for sampling.

    Returns:
        ndarray, ndarray: Samples from X as flattened array, Sampled/sparse 2D-Array of X.
    """    
    b = X.flat[idx]

    mask = np.zeros(X.shape)
    mask.flat[idx] = 255
    Xm = 0 * np.ones(X.shape)
    Xm.flat[idx] = X.flat[idx]

    return b, Xm

def generateCloudFile(filename, center, size, points, idx, angle=0, displayGrid=True):
    """Creates a Nanonis grid experimental file (*.ngef2) with a cloud of measurement points and a JSON file with the parameters of the sparse grid.

    Args:
        filename (str): Path and filename for the Nanonis and JSON file without file extension.
        center (tuple): Center position of the grid in meters: (x, y) 
        size (tuple): Width and height of the grid in meters: (width, height)
        points (tuple): Number of points in x- and y-direction: (nx, ny)
        idx (ndarray): Random indices (1D) for sampling.
        angle (float, optional): Rotation angle of the grid in degrees. Positive values correspond to counterclockwise rotations. Defaults to 0.
        displayGrid (bool, optional): Displays the sparse grid if True. Defaults to True.

    Returns:
        ndarray, ndarray, ndarray: x coordinates for each pixel, y coordinates for each pixel, Measurement points for each pixel (1 - True, 0 - False).
    """
    xc, yc = center
    width, height = size
    nx, ny = points
    x = np.linspace(-width/2, width/2, nx)
    y = np.linspace(height/2, -height/2, ny)
    theta = angle * (2*np.pi)/360

    Xcoord = np.zeros((ny, nx))
    Ycoord = np.zeros((ny, nx))
    MeasPoint = np.zeros((ny, nx))
    DispPoint = np.zeros((ny, nx))
    MeasPoint.flat[idx] = 1

    nDisplayMax = 1000
    if len(idx) <= nDisplayMax:
        nDisplayMax = len(idx)
    idxDisp = np.random.choice(idx, nDisplayMax, replace=False)
    idxDisp = np.sort(idxDisp) 
    DispPoint.flat[idxDisp] = 1

    for j in range(len(y)):
        for i in range(len(x)):
            if angle != 0:
                Xcoord[j,i] = x[i] * np.cos(theta) - y[j] * np.sin(theta) + xc
                Ycoord[j,i] = x[i] * np.sin(theta) + y[j] * np.cos(theta) + yc
            else:
                Xcoord[j,i] = x[i]
                Ycoord[j,i] = y[j]

    fileNGEF = filename + '.ngef2'
    with open(fileNGEF, 'w') as f:
        for i in range(nx*ny):
            if MeasPoint.flat[i] == 1:
                line = '{}\t{}\t{}\t{}\n'.format(Xcoord.flat[i], Ycoord.flat[i], int(DispPoint.flat[i]), int(MeasPoint.flat[i]))
                f.write(line)

    d = {}
    d['nx'] = nx
    d['ny'] = ny
    d['xc (m)'] = xc
    d['yc (m)'] = yc
    d['width (m)'] = width
    d['height (m)'] = height
    d['angle (deg)'] = angle
    d['idx'] = idx.tolist()
    fileJSON = filename + '.json'
    with open(fileJSON, 'w', encoding ='utf8') as json_file:
        json.dump(d, json_file, allow_nan=True, indent=4)

    if displayGrid:
        plt.figure(figsize=(8,8))
        plt.gca().set_aspect('equal')
        for j in range(ny):
            for i in range(nx):
                if MeasPoint[j,i] == 1:
                    plt.plot(Xcoord[j,i], Ycoord[j,i], '.', c='black')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()

    return Xcoord, Ycoord, MeasPoint