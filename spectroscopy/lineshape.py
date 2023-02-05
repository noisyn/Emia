# Copyright (c) 2023 Taner Esat <t.esat@fz-juelich.de>

import numpy as np

def gaussian(x, A, x0, gamma, offset):
    """Calculates Gaussian function.

    Args:
        x (ndarray): x-values.
        A (float): Amplitude of Gaussian function.
        x0 (float): Position of Gaussian function.
        gamma (float): Full-width-haf-maximum (FWHM).
        offset (float): Offset.

    Returns:
        ndarray: Gaussian function.
    """    
    sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    y = A * np.exp((-1/2) * np.square(x-x0)/(np.square(sigma))) + offset
    return y

def lorentzian(x, A, x0, gamma, offset):
    """Calculates Lorentzian function.

    Args:
        x (ndarray): x-values.
        A (float): Amplitude of Lorentzian function.
        x0 (float): Position of Lorentzian function.
        gamma (float): Full-width-haf-maximum (FWHM).
        offset (float): Offset.

    Returns:
        ndarray: Lorentzian function.
    """    
    y = (0.5 * gamma) / (np.square(x - x0) + np.square(0.5 * gamma))
    y = A * (y / np.max(y)) + offset
    return y

def fano(x, A, x0, gamma, q, offset):
    """Calculates Fano resonance.

    Args:
        x (ndarray): x-values.
        A (float): Amplitude of Fano resonance.
        x0 (float): Position of Fano resonance.
        gamma (float): Full-width-haf-maximum (FWHM).
        q (float): Fanor factor. Values between 0 anf +inf.
        offset (float): Offset.

    Returns:
        ndarray: Fano resonance.
    """    
    eps = (x - x0) / (gamma / 2)
    y = np.square(q + eps) / (1 + np.square(eps))
    y = A * (y / np.max(y)) + offset
    return y

def frota(x, A, x0, gamma_k, phi, offset):
    """Calculates Frota function.

    Args:
        x (ndarray): x-values.
        A (float): Amplitude of Frota function.
        x0 (float): Position of Frota function.
        gamma_k (float): Width -> proportional to Kondo temperature Tk = (Gamma_K * 2 * pi * 0.103) / kB with Boltzmann constant kB.
        phi (float): Form factor. Similar role as q for the Fano line shape.
        offset (float): Offset.

    Returns:
        ndarray: Frota function.
    """    
    # y = np.real(np.sqrt((1j*gamma_k) / (x - x0 + 1j*gamma_k)))
    y = np.imag((1j * np.exp(1j * phi)) * np.sqrt((1j*gamma_k) / (x - x0 + 1j*gamma_k)))
    y = A * (y / np.max(y)) + offset
    return y