# Copyright (c) 2023 Taner Esat <t.esat@fz-juelich.de>


import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy import signal, integrate, interpolate
from scipy.optimize import curve_fit

class PE():
    def __init__(self, energyRange=[-5e-3, 5e-3], points=2**18, charge=1):
        # Initialize energy ranges
        self.hbar_omega = np.linspace(energyRange[0], energyRange[1], points)
        self.E = np.linspace(energyRange[0], energyRange[1], points)
        self.ncharge = charge

        # Initialize physical constants
        self.kB = constants.physical_constants['Boltzmann constant in eV/K'][0]
        self.q = constants.physical_constants['atomic unit of charge'][0]
        self.h = constants.physical_constants['Planck constant'][0]
        self.hbar = constants.physical_constants['Planck constant in eV/Hz'][0] / (2 * np.pi)
        self.R_Q = self.h / (2 * np.power(self.q, 2))

    def impedance_Z(self, hbar_omega, hbar_omega_0, alpha, R_env):
        Z = R_env * (1 + (1j/alpha) * np.tan((np.pi/2) * (hbar_omega / hbar_omega_0))) / (1 + (1j*alpha) * np.tan((np.pi/2) * (hbar_omega / hbar_omega_0)))
        return Z

    def total_impedance(self, hbar_omega, C_J, hbar_omega_0, alpha, R_env):
        Z_T = 1 / ((1j * hbar_omega / self.hbar * C_J) + np.power(self.impedance_Z(hbar_omega, hbar_omega_0, alpha, R_env), -1))
        return Z_T

    def PN(self, C_J, T): 
        E_C = np.power(self.q * self.ncharge, 2) / (2 * C_J) * constants.physical_constants['joule-electron volt relationship'][0] # in eV
        P_N = 1 / np.sqrt(4 * np.pi * E_C * self.kB * T ) * np.exp((-1) * np.power(self.E, 2) / (4 * E_C * self.kB * T))
        return P_N

    def inhomogenity_I(self, T, C_J, hbar_omega_0, alpha, R_env):
        beta = 1 / (self.kB * T)
        Z_T0 = self.total_impedance(0, C_J, hbar_omega_0, alpha, R_env)
        D = (np.pi / beta) * np.real(Z_T0) / self.R_Q
        I = (1 / np.pi) * D / (np.power(D, 2) + np.power(self.E, 2))
        return I

    def function_k(self, C_J, hbar_omega_0, alpha, T, R_env):
        beta = 1 / (self.kB * T)
        Z_T = self.total_impedance(self.hbar_omega, C_J, hbar_omega_0, alpha, R_env)
        Z_T0 = self.total_impedance(0, C_J, hbar_omega_0, alpha, R_env) 
        k = (1 / (1 - np.exp((-1) * beta * self.hbar_omega))) * (np.real(Z_T) / self.R_Q) - (1 / (beta *  self.hbar_omega) *  (np.real(Z_T0) / self.R_Q))
        return k

    def function_kappa(self, C_J, hbar_omega_0, alpha, T, R_env):
        beta = 1 / (self.kB * T)
        Z_T = self.total_impedance(self.hbar_omega, C_J, hbar_omega_0, alpha, R_env)
        hbar_nu_n = lambda n: 2 * n * np.pi / beta

        matsubara_sum = 0
        for n in range(1, 101):
            hbar_vn = hbar_nu_n(n)
            matsubara_sum += (hbar_vn / (np.power(hbar_vn, 2) + np.power(self.hbar_omega, 2))) * (self.total_impedance(((-1j)*hbar_vn), C_J, hbar_omega_0, alpha, R_env) / self.R_Q)

        kappa = ((-1) / (1 - np.exp((-1) * beta * self.hbar_omega))) * (np.imag(Z_T) / self.R_Q) - (2 / beta) * matsubara_sum
        return kappa

    def kernel_K(self, hbar_omega, C_J, hbar_omega_0, alpha, T, R_env):
        k = self.function_k(hbar_omega, C_J, hbar_omega_0, alpha, T, R_env)
        kappa = self.function_kappa(hbar_omega, C_J, hbar_omega_0, alpha, T, R_env)

        beta = 1 / (self.kB * T)
        Z_T0 = self.total_impedance(0, C_J, hbar_omega_0, alpha, R_env) 
        D = (np.pi / beta) * np.real(Z_T0) / self.R_Q

        kernel = ((1 * self.E) / (np.power(D, 2) + np.power(self.E, 2))) * k + ((1 * D) / (np.power(D, 2) + np.power(self.E, 2))) * kappa

        return kernel

    def P0(self, C_J, hbar_omega_0, alpha, T, R_env):
        k = self.function_k(C_J, hbar_omega_0, alpha, T, R_env)
        kappa = self.function_kappa(C_J, hbar_omega_0, alpha, T, R_env)

        beta = 1 / (self.kB * T)
        Z_T0 = self.total_impedance(0, C_J, hbar_omega_0, alpha, R_env) 
        D = (np.pi / beta) * np.real(Z_T0) / self.R_Q
        Inh = self.inhomogenity_I(T, C_J, hbar_omega_0, alpha, R_env)
        P_0 = Inh
        P_0 = P_0 / np.sum(P_0)

        for m in range(5):
            intPart_k = signal.fftconvolve(k, P_0, mode='same')
            intPart_kappa = signal.fftconvolve(kappa, P_0, mode='same')
            intpart_total = (self.E / (np.power(D, 2) + np.power(self.E, 2))) * intPart_k + (D / (np.power(D, 2) + np.power(self.E, 2))) * intPart_kappa
            P_0 = Inh + intpart_total
            P_0 = P_0 / np.sum(P_0) 

        return P_0

    def PE(self, C_J, hbar_omega_0, alpha, T, R_env):
        P_0 = self.P0(C_J, hbar_omega_0, alpha, T, R_env)
        P_N = self.PN(C_J, T)
        P_E = signal.fftconvolve(P_0, P_N, mode='same')
        P_E = P_E / integrate.trapz(P_E, self.E)
        return P_E

    def gamma(self, C_J, hbar_omega_0, alpha, T, R_env):
        pre_factor = 1 
        beta = 1 / (self.kB * T)
        # int_fermi = self.E / (1 - np.exp((-1) * beta * self.E)) # old and wrong
        int_fermi = 1 / (1 + np.exp((-1) * beta * self.E))
        P_E = self.PE(C_J, hbar_omega_0, alpha, T, R_env)
        # convolution with Fermi --- 1x or 2x ???
        G = signal.fftconvolve(int_fermi, P_E, mode='same')
        G = pre_factor * signal.fftconvolve(int_fermi, G, mode='same')

        return G

    def current(self, C_J, hbar_omega_0, alpha, T, R_env):
        G_r = self.gamma(C_J, hbar_omega_0, alpha, T, R_env)
        G_l = np.flip(G_r)
        I = (G_r - G_l) * self.q
        return self.E, I

    def fit_PE(self, x, R_env, alpha, hbar_omega_0, T, C_J, offset):
        points = 2**18
        hbar_omega = np.linspace(-5e-3, 5e-3, points)
        E = np.linspace(-5e-3, 5e-3, points)

        E_C = np.power(self.q, 2) / (2 * C_J) * 6.241509e18

        I = self.current(E, hbar_omega, C_J, E_C, hbar_omega_0, alpha, T, R_env)
        dIdV = np.gradient(I, E) 
        dIdV = dIdV/dIdV[int(len(E)/4)] + offset

        interpolated_dIdV = interpolate.interp1d(E, dIdV)

        return interpolated_dIdV(x).astype(float)
