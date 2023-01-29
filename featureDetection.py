# Copyright (c) 2023 Taner Esat <t.esat@fz-juelich.de>

import warnings

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", "Mean of empty slice")

class FDSTS():
    def __init__(self, x_table, specgrid, smoothing_param=dict(wl=89, po=2), xlabel='Bias voltage (V)'):
        """Initializes the parameters for the feature detection algorithm. Smoothing of the spectra is done by a Savitzky-Golay filter.

        Args:
            x_table (ndarray): X values.
            specgrid (ndarray): Specgrid data in format [y, x, spectrum].
            smoothing_param (dict, optional): Parameters for smoothing: wl (int): The length of the filter window. po (int): The order of the polynomial used to fit the samples. po must be less than wl. Defaults to dict(wl=89, po=2).
            xlabel (str, optional): Label of x-axis for displaying. Defaults to 'Bias voltage (mV)'.
        """
        self.x_table = x_table
        self.specgrid = specgrid
        self.xlabel = xlabel
        self.smoothing_param = smoothing_param
        self.ny = self.specgrid.shape[0]
        self.nx = self.specgrid.shape[1]
        self.points = self.specgrid.shape[2]

        # Copy spectra
        self.norm_spectrum = np.copy(self.specgrid)
        self.orig_spectrum = np.copy(self.norm_spectrum)

        # Smooth and normalize spectra between [0,1]
        self.sm_specgrid = np.zeros(shape=(self.ny, self.nx, self.points))
        for x in range(0, self.nx):
            for y in range(0, self.ny):    
                # Normalize
                sm_spectrum = self.smoothSpectrum(self.norm_spectrum[y, x], self.smoothing_param['wl'], self.smoothing_param['po'])
                self.norm_spectrum[y, x] = (self.norm_spectrum[y, x] - np.min(sm_spectrum))/np.max(sm_spectrum) 
                # Smooth
                sm_spectrum = self.smoothSpectrum(self.norm_spectrum[y, x], self.smoothing_param['wl'], self.smoothing_param['po'])
                self.sm_specgrid[y, x] = sm_spectrum

    def fit(self, peak_threshold, shoulder_threshold):
        """Detects all features in the specgrid.

        Args:
            peak_threshold (float): Threshold/weight for detecting peaks.
            shoulder_threshold (float): Threshold/weight for detecting shoulders.
        """              
        self.feature_map_intensity = np.zeros(shape=(self.ny, self.nx, self.points))
        self.feature_map_xpos = np.zeros(shape=(self.ny, self.nx, self.points))

        self.deriv1_specgrid = np.zeros(shape=(self.ny, self.nx, self.points))
        self.deriv2_specgrid = np.zeros(shape=(self.ny, self.nx, self.points))

        self.feature_counts = np.zeros(shape=(self.points))

        # Find features (peaks)
        for x in range(0, self.nx):
            for y in range(0, self.ny):          
                sm_spectrum = self.sm_specgrid[y, x]
                
                deriv1_spec = self.smoothSpectrum(np.gradient(sm_spectrum), self.smoothing_param['wl'], self.smoothing_param['po'])
                deriv2_spec = self.smoothSpectrum(np.gradient(deriv1_spec), self.smoothing_param['wl'], self.smoothing_param['po'])

                self.deriv1_specgrid[y, x] = deriv1_spec
                self.deriv2_specgrid[y, x] = deriv2_spec

                feature_indices = self.detectFeatures(sm_spectrum, deriv1_spec, deriv2_spec, peak_threshold, shoulder_threshold)
                self.feature_counts[feature_indices] += 1

                features = np.zeros(shape=(self.points))
                features[:] = np.NaN
                features[feature_indices] = self.orig_spectrum[y, x, feature_indices]
                self.feature_map_intensity[y, x] = features

                features[:] = np.NaN
                features[feature_indices] = self.x_table[feature_indices]
                self.feature_map_xpos[y, x] = features

    def smoothSpectrum(self, spec, wl, po):
        """Smoothes the spectrum using a Savitzky-Golay filter.

        Args:
            spec (ndarray): Spectrum.
            wl (int): The length of the filter window.
            po (int): The order of the polynomial used to fit the samples. po must be less than wl.

        Returns:
            ndarray: Smoothed spectrum.
        """        
        return savgol_filter(spec, window_length=wl, polyorder=po)
    
    def detectFeatures(self, sm_spectrum, deriv1_spec, deriv2_spec, peak_threshold, shoulder_threshold):
        """Detects features in an individual spectrum.

        Args:
            sm_spectrum (ndarray): Smoothed spectrum.
            deriv1_spec (ndarray): First derivative of the spectrum.
            deriv2_spec (ndarray): Second derivative of the spectrum.
            peak_threshold (float): Threshold/weight for detecting peaks.
            shoulder_threshold (float): Threshold/weight for detecting shoulders.

        Returns:
            list: Indices of the detected features.
        """        
        # Sign change in derivatives:
        # 0 no sign change -> no extrema
        # +1 sign change from - to + -> minima
        # -1 sign change from + to - -> maxima
        sgn_change_deriv1 = np.gradient(np.sign(deriv1_spec))
        sgn_change_deriv2 = np.gradient(np.sign(deriv2_spec))

        # Indices of maxima/minima and saddle points
        maxmin_idx = np.argwhere(sgn_change_deriv1)
        saddle_idx = np.argwhere(sgn_change_deriv2)
        all_extrema_idx = np.vstack((maxmin_idx, saddle_idx))
        all_extrema_idx = np.sort(all_extrema_idx, axis=None)

        feature_idx = []
        # weight = 0
        for x in range(len(sm_spectrum)):
            # Maxima
            if sgn_change_deriv1[x] < 0:
                extrema_idx = np.argwhere(all_extrema_idx == x)[0][0]
                if extrema_idx-2 > 0 and extrema_idx+2 < len(all_extrema_idx):
                    weighting_idx_left = all_extrema_idx[extrema_idx-2]
                    weighting_idx_right = all_extrema_idx[extrema_idx+2]
                    extrema_weight = np.abs(deriv1_spec[weighting_idx_left]) + np.abs(deriv1_spec[weighting_idx_right])
                    if extrema_weight > peak_threshold:
                        feature_idx.append(x)
                        # weight = extrema_weight
            # Saddle points
            elif sgn_change_deriv2[x] != 0 and sgn_change_deriv1[x] == 0:
                extrema_idx = np.argwhere(all_extrema_idx == x)[0][0]
                if extrema_idx-2 > 0 and extrema_idx+2 < len(all_extrema_idx):
                    weighting_idx_left = all_extrema_idx[extrema_idx-2]
                    weighting_idx_right = all_extrema_idx[extrema_idx+2]
                    shoulder_weight_left = (np.abs(deriv1_spec[weighting_idx_left]) - np.abs(deriv1_spec[x]))
                    shoulder_weight_right = (np.abs(deriv1_spec[weighting_idx_right]) - np.abs(deriv1_spec[x]))
                    if shoulder_weight_left > shoulder_threshold and shoulder_weight_right > shoulder_threshold:
                        feature_idx.append(x)
                        # weight = (shoulder_weight_left + shoulder_weight_right) / 2

        return feature_idx

    def getFeatureMap(self, xmin, xmax, type='pos'):
        """Returns the 2D feature map in the interval [xmin, xmax].

        Args:
            xmin (float): Lower bound.
            xmax (float): Upper bound.
            type (str, optional): 'int': Intensity of features, 'pos': Position of features. Defaults to 'pos'.

        Returns:
            ndarray: 2D feature map. Either intensities or positions of features are returned.
        """        
        min_idx = np.abs(self.x_table-xmin).argmin()
        max_idx = np.abs(self.x_table-xmax).argmin()
        if min_idx > max_idx:
            tmp = min_idx
            min_idx = max_idx
            max_idx = tmp
        
        if type == 'int':
            fd_map = np.nanmean(self.feature_map_intensity[:,:,min_idx:max_idx], axis=2)
        elif type == 'pos':
            fd_map = np.nanmean(self.feature_map_xpos[:,:,min_idx:max_idx], axis=2)

        return fd_map

    def getSpectra(self, x, y):
        """Returns spectra at position (x, y).

        Args:
            x (int): Coordinate in x-direction.
            y (int): Coordinate in y-direction.

        Returns:
            ndarrays: X values, Original spectrum, Smoothed spectrum, First derivative of spectrum, Second derivative of spectrum.
        """        
        norm_spectrum = self.norm_spectrum[y, x, :]
        sm_spectrum = self.sm_specgrid[y, x, :]
        deriv1_spectrum = self.deriv1_specgrid[y, x, :]
        deriv2_spectrum = self.deriv2_specgrid[y, x, :]

        return self.x_table, norm_spectrum, sm_spectrum, deriv1_spectrum, deriv2_spectrum
    
    def getFeatures(self, x, y):
        """Returns the indices of the detected features at position (x, y).

        Args:
            x (int): Coordinate in x-direction.
            y (int): Coordinate in y-direction.

        Returns:
            ndarray, ndarray: X values, Indices of detected features.
        """        
        feature_idx = np.where(self.feature_map_intensity[y, x, :] > 0)
        return self.x_table, feature_idx

    def getHistogram(self):
        """Returns histogram of detected features.

        Returns:
            ndarray, ndarray: X values, Counts of detected features at each x value.
        """        
        return self.x_table, self.feature_counts

    def showHistogram(self):
        """Displays the histogram of the detected features.
        """        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x_table, y=self.feature_counts, 
                        mode='markers', name='Detected features'))
        fig.update_layout(title='Feature histogram', 
                            xaxis_title=self.xlabel, yaxis_title='Counts (#)')
        fig.show()

    def showFeatureMap(self, xmin, xmax, mode='dual', zmin=0, zmax=0):
        """Displays the feature map in the interval [xmin, xmax].

        Args:
            xmin (float): Lower bound.
            xmax (float): Upper bound.
            mode (str, optional): Display mode. 'int': Intensity of features, 'pos': Position of features, 'dual': Intensity and position of features. Defaults to 'dual'.
            zmin (float, optional): Lower bound of color domain. Defaults to 0.
            zmax (float, optional): Upper bound of color domain. Defaults to 0.
        """        
        if mode == 'int':
            fd_map = self.getFeatureMap(xmin, xmax, type='int')
            fd_label = 'Intensity (a.u.)'
        elif mode == 'pos':
            fd_map = self.getFeatureMap(xmin, xmax, type='pos')
            fd_label = self.xlabel
        else:
            fd_map_int = self.getFeatureMap(xmin, xmax, type='int')
            fd_map_xpos = self.getFeatureMap(xmin, xmax, type='pos')
            fd_label_int = 'Intensity (a.u.)'
            fd_label_xpos = self.xlabel

        if mode == 'dual':
            fig = make_subplots(rows=1, cols=5, shared_yaxes=True,
                                subplot_titles=('Feature intensities', '', 'Feature positions'),
                                specs=[[{"colspan": 2}, None, {"colspan": 1}, {"colspan": 2}, None]])
            if zmin == 0 and zmax == 0:
                fig.add_trace(go.Heatmap(z=fd_map_int, 
                                colorbar=dict(title=dict(text=fd_label_int, side='right'), 
                                thickness=10, x=0.395)), row=1, col=1)         
            else:
                fig.add_trace(go.Heatmap(z=fd_map_int, zmin=zmin, zmax=zmax,
                                colorbar=dict(title=dict(text=fd_label_int, side='right'), 
                                thickness=10, x=0.395)), row=1, col=1)              
            fig.add_trace(go.Heatmap(z=fd_map_xpos, 
                            colorbar=dict(title=dict(text=fd_label_xpos, side='right'),
                            thickness=10)), row=1, col=4)
            fig.update_yaxes(title_text='Y', autorange='reversed', row=1, col=1) 
            fig.update_yaxes(title_text='Y', autorange='reversed', row=1, col=4)  
            fig.update_xaxes(title_text='X', row=1, col=1) 
            fig.update_xaxes(title_text='X', row=1, col=4) 
            fig.update_layout(title="Feature map: {} V to {} V".format(xmin, xmax), 
                                width=800, height=400, yaxis_scaleanchor="x")
            fig.show()
        else:
            fig = go.Figure()
            if zmin == 0 and zmax == 0:
                fig.add_trace(go.Heatmap(z=fd_map, colorbar=dict(title=dict(text=fd_label, side='right'))))
            else:
                fig.add_trace(go.Heatmap(z=fd_map, zmin=zmin, zmax=zmax, colorbar=dict(title=dict(text=fd_label, side='right'))))
            fig.update_layout(title="Feature map: {} V to {} V".format(xmin, xmax), 
                                xaxis_title='X', yaxis_title='Y',
                                width=450, height=450,
                                yaxis=dict(autorange='reversed'), yaxis_scaleanchor="x")
            fig.show()

    def showSpectrum(self, x, y):
        """Displays the spectrum at position (x, y).

        Args:
            x (int): Coordinate in x-direction.
            y (int): Coordinate in y-direction.
        """        
        _, norm_spectrum, sm_spectrum, deriv1_spectrum, deriv2_spectrum = self.getSpectra(x, y)
        _, feature_idx = self.getFeatures(x, y)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.x_table, y=norm_spectrum, 
                        mode='lines+markers', name='dI/dV'))
        fig.add_trace(go.Scatter(x=self.x_table, y=sm_spectrum, 
                        mode='lines', name='Smoothed'))
        fig.add_trace(go.Scatter(x=self.x_table[feature_idx], y=sm_spectrum[feature_idx], 
                        mode='markers', name='Detected features'))
        fig.add_trace(go.Scatter(x=self.x_table, y=deriv1_spectrum,
                        mode='lines', name='1st derivative'))
        fig.add_trace(go.Scatter(x=self.x_table, y=deriv2_spectrum,
                        mode='lines', name='2nd derivative'))
        fig.update_layout(title="Spectrum at ({}, {})".format(x, y), 
                            xaxis_title=self.xlabel, yaxis_title='Signal (a.u.)')
        fig.show()