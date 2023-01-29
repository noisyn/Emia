# Copyright (c) 2023 Taner Esat <t.esat@fz-juelich.de>

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve, savgol_filter


def ArcsineDistribution(x, V):
    """Arcsine distribution: Broadening due to sinusoidal signals.

    Args:
        x (ndarray): Bias voltage / energy range.
        V (ndarray): Amplitude of sinusoidal signal (peak-to-zero).

    Returns:
        ndarray: Arcsine distribution.
    """    
    f = np.zeros(len(x))
    for i in range(len(x)):
        if np.abs(x[i]) < V:
            f[i] = (1/(np.pi*V)) * 1/(np.sqrt(1 - np.power((x[i]/V), 2)))
        else:
            f[i] = 0
    return f

def estimateRFBroadening(bias, dIdV, dIdVRF, fitRange, Vrf0, displayPlot=True):
    """Calculates the amplitude of a radio-frequency signal from dI/dV spectra.

    Args:
        bias (ndarray): Bias voltage in Volts.
        dIdV (ndarray): dI/dV spectrum with RF signal OFF.
        dIdVRF (ndarray): dI/dV spectrum with RF signal ON.
        fitRange (list): Lower and upper limits for fitting range, e.g. [-75e-3, -50e-3]
        Vrf0 (float): Estimate of RF amplitude.
        displayPlot (bool, optional): Displays the spectra and fit if True. Defaults to True.

    Returns:
        float: Estimated RF amplitude.
    """    
    llim = fitRange[0]
    ulim = fitRange[1]
    
    dIdV_smoothed = savgol_filter(dIdV, 50, 3)
    dIdVRF_smoothed = savgol_filter(dIdVRF, 50, 3)

    xarc = np.linspace(-500e-3, 500e-3, 1000)
    dIdVinterp = np.interp(xarc, bias, dIdV_smoothed)
    dIdVRFinterp = np.interp(xarc, bias, dIdVRF_smoothed)

    def fitArcsineDistribution(x, Vrf, A, offset):
        nonlocal xarc
        nonlocal dIdVinterp
        arcDist = ArcsineDistribution(xarc, Vrf)
        result = fftconvolve(arcDist, dIdVinterp, mode='same') 
        result = np.interp(x, xarc, result)
        return A*result + offset

    xfit = np.linspace(llim, ulim, 1000)
    dIdVRFfit = np.interp(xfit, xarc, dIdVRFinterp)
    popt, pcov = curve_fit(fitArcsineDistribution, xfit, dIdVRFfit, [Vrf0, 1, 0])
    perr = np.sqrt(np.diag(pcov))
    Vrf = popt[0]
    # print(popt)

    if displayPlot:
        plt.figure()
        plt.plot(bias, dIdV, '.', label='RF off')
        plt.plot(bias, dIdVRF, '.', label='RF on')
        plt.plot(bias, fitArcsineDistribution(bias, *popt), c='black', label='Vrf = {:.1f} mV'.format(Vrf/1e-3))
        plt.xlim(llim, ulim)
        plt.xlabel('Bias Voltage (V)')
        plt.ylabel('dI/dV (a.u.)')
        plt.legend()
        plt.show()

    return Vrf

def LockinModulationBroadeningFunction(x, V):
    """Lock-In modulation broadening function.

    Args:
        x (ndarray): Bias voltage / energy range.
        V (ndarray): Lock-in modulation amplitude (peak-to-zero).

    Returns:
        ndarray: Lock-In broadening function.
    """    
    f = np.zeros(len(x))
    for i in range(len(x)):
        if np.abs(x[i]) <= V:
            f[i] = (2*np.sqrt(np.power(V, 2) - np.power(x[i],2)))/(np.pi * np.power(V, 2))
        else:
            f[i] = 0
    return f

def FermiDiracDistributionDerivative(x, T):
    """Derivative of Fermi-Dirac distribution.

    Args:
        x (ndarray): Bias voltage / energy range.
        T (float): Temperature in Kelvin.

    Returns:
        ndarray: Derivative of Fermi-Dirac distribution.
    """    
    kB = constants.physical_constants['Boltzmann constant in eV/K'][0]
    beta = 1/(kB * T)
    f = beta / (4 * np.power(np.cosh(beta*x / 2), 2))
    return f

def calculateConvolution(f1, f2):
    """Calculates the convolution of f1 and f2.

    Args:
        f1 (ndarray): Array / Signal #1.
        f2 (ndarray): Array / Signal #2.

    Returns:
        ndarray: Convoluted signal.
    """    
    conv = fftconvolve(f1, f2, mode="same")
    return conv

def temperatureBroadeningDOS(bias, y, T):
    """Calculates the temperature broadending of y (density of states).

    Args:
        bias (ndarray): Bias voltage.
        y (ndarray): Array / Signal.
        T (float): Temperature in Kelvin.

    Returns:
        ndarray: Temperature broadened y.
    """    
    return calculateConvolution(y, FermiDiracDistributionDerivative(bias, T))

def modulationBroadeningDOS(bias, y, V):
    """Calculates the broadending of y (density of states) due to Lock-In modulation.

    Args:
        bias (ndarray): Bias voltage.
        y (ndarray): Array / Signal.
        V (float): Lock-In modulation amplitude (peak-to-zero).

    Returns:
        ndarray: Lock-In modulation broadened y.
    """    
    return calculateConvolution(y, LockinModulationBroadeningFunction(bias, V))

def RFBroadeningDOS(bias, y, V):
    """Calculates the broadending of y (density of states) due to radio-frequency (RF) signal.

    Args:
        bias (ndarray): Bias voltage.
        y (ndarray): Array / Signal.
        V (float): Amplitude of sinusoidal signal (peak-to-zero).

    Returns:
        ndarray: RF broadened y.
    """    
    return calculateConvolution(y, ArcsineDistribution(bias, V))