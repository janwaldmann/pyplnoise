# -*- coding: utf-8 -*-
"""Noise generators for long streams of 1/f^alpha noise.

The time domain methods used follow Stephane Plaszczynski, Fluct. Noise
Lett. 7: R-R13, 2007. DOI: 10.1142/S0219477507003635, see also
https://arxiv.org/abs/astro-ph/0510081. The code also sticks to the
coefficient notation in this article (which is different to the SciPy docs).
"""

import numpy as np
from scipy import signal


class WhiteNoise:
    """White noise generator (constant power spectrum)."""

    def __init__(self, f_sample: float, psd: float = 1.0):
        """Create a WhiteNoise instance.

        Args:
            f_sample (float): sampling frequency in Hz
            psd (float): constant value of the two-sided power spectral density.
                The RMS value of the generated noise is sqrt(f_sample x psd).

        """
        self.fs = f_sample
        self.rms = np.sqrt(f_sample * psd)
        self.rng = np.random.default_rng()

    def get_sample(self) -> float:
        """Retrieve a single sample."""
        return self.rng.normal(loc=0.0, scale=self.rms)

    def get_series(self, npts: int) -> np.ndarray:
        """Retrieve an array of npts samples."""
        if npts > np.iinfo(int).max:
            raise ValueError("""Argument 'npts' must be an integer <= {}. If you
            want to obtain more samples, run get_series() several times 
            and concatenate the results.""".format(np.iinfo(int).max))
        return self.rng.normal(loc=0.0, scale=self.rms, size=npts)


class RedNoise:
    """Red (Brownian) noise generator (1/f^2 power spectrum).

    The two-sided noise PSD is scaled such that PSD(f = 1 Hz) = 1. Below f_min,
    the noise is white.
    """

    def __init__(self, f_sample: float, f_min: float, init_filter: bool = True):
        """Create a RedNoise instance.

        Args:
            f_sample (float): sampling frequency in Hz
            f_min (float): frequency cutoff in Hz. Below f_min, the noise will be
                white.
            init_filter (bool, optional): settle filter during object initialization.
                This might take some time depending on the magnitude of the ratio
                f_sample/f_min. Default and highly recommended: True.

        """
        self.fs = f_sample
        self.fmin = f_min
        self.whitenoise = WhiteNoise(self.fs, psd=1.0)
        self.scaling = 1. / (self.fs * self.fmin)
        self.a = np.array([2. * np.pi * self.fmin])
        self.b = np.array(
            [1.0, -1.0 * np.exp(-2. * np.pi * self.fmin / self.fs)])
        self.zi = signal.lfilter_zi(
            self.a, self.b) * self.whitenoise.get_sample()
        if init_filter:
            npts_req = np.ceil(2. * self.fs / self.fmin)
            # safeguard for machines with small memory
            npts_per_run_max = np.iinfo(int).max // 16
            if npts_req > npts_per_run_max:
                for i in range(np.ceil(npts_req / npts_per_run_max).astype(int)):
                    _ = self.get_series(npts_per_run_max)
            else:
                _ = self.get_series(int(npts_req))

    def get_sample(self) -> float:
        """Retrieve a single sample."""
        sample, self.zi = signal.lfilter(
            self.a, self.b, self.whitenoise.get_sample(), zi=self.zi)
        return sample * self.scaling

    def get_series(self, npts: int) -> np.ndarray:
        """Retrieve an array of npts samples."""
        if npts > np.iinfo(int).max:
            raise ValueError("""Argument 'npts' must be an integer <= {}. If you
            want to obtain more samples, run get_series() several times 
            and concatenate the results.""".format(np.iinfo(int).max))
        samples, self.zi = signal.lfilter(
            self.a, self.b, self.whitenoise.get_series(npts), zi=self.zi)
        return samples * self.scaling


class AlphaNoise:
    """Colored noise noise generator (arbitrary 1/f^alpha power spectrum).

    The two-sided noise PSD is scaled such that PSD(f = 1 Hz) = 1. The noise
    generator has user-specified lower and upper cutoff frequencies. Below and above
    these frequencies, the generated noise is white.
    """

    def __init__(self, f_sample: float, f_min: float, f_max: float,
                 alpha: float, init_filter: bool = True):
        """Create an AlphaNoise instance.

        Args:
            f_sample (float): sampling frequency in Hz
            f_min (float): lower frequency cutoff in Hz. Below f_min, the noise
                will be white.
            f_max (float): upper frequency cutoff in Hz. Above f_max, the noise
                will be white.
            alpha (float): exponent of the 1/f^alpha power spectrum. Must be in the
                interval [0.01, 2.0].
            init_filter (bool, optional): settle filter during object initialization.
                This might take some time depending on the magnitude of the ratio
                f_sample/f_min. Default and highly recommended: True.

        """
        if alpha > 2. or alpha < 0.01:
            raise ValueError(
                "The exponent must be in the range 0.01 <= alpha <= 2.")
        self.fs = f_sample
        self.fmin = f_min
        self.fmax = f_max
        self.alpha = alpha
        self.whitenoise = WhiteNoise(self.fs, psd=1.0)
        log_w_min = np.log10(2. * np.pi * self.fmin)
        log_w_max = np.log10(2. * np.pi * self.fmax)
        self.num_spectra = np.ceil(4.5 * (log_w_max - log_w_min)).astype(int)
        dp = (log_w_max - log_w_min) / self.num_spectra
        self.a = [None] * self.num_spectra
        self.b = [None] * self.num_spectra
        self.zi = [None] * self.num_spectra
        for i in range(0, self.num_spectra):
            log_p_i = log_w_min + dp * 0.5 * ((2. * i + 1.) - self.alpha / 2.)
            filter_f_min = np.power(10., log_p_i) / (2. * np.pi)
            filter_f_max = np.power(
                10., log_p_i + (dp * self.alpha / 2.)) / (2. * np.pi)
            if i == 0:
                self.fmin = filter_f_min
            a0, a1, b1 = self._calc_filter_coeff(filter_f_min, filter_f_max)
            self.a[i] = np.array([a0, a1])
            self.b[i] = np.array([1.0, -1.0 * b1])
            self.zi[i] = signal.lfilter_zi(self.a[i], self.b[i])
        self.fmax = filter_f_max
        self.scaling = 1. / np.power(self.fmax, alpha / 2.)
        if init_filter:
            npts_req = np.ceil(2. * self.fs / self.fmin)
            # safeguard for machines with small memory
            npts_per_run_max = np.iinfo(int).max // 16
            if npts_req > npts_per_run_max:
                for i in range(np.ceil(npts_req / npts_per_run_max).astype(int)):
                    _ = self.get_series(npts_per_run_max)
            else:
                _ = self.get_series(int(npts_req))

    def get_sample(self) -> float:
        """Retrieve a single sample."""
        sample = self.whitenoise.get_sample()
        for i in range(0, len(self.a)):
            sample, self.zi[i] = signal.lfilter(
                self.a[i], self.b[i], sample, zi=self.zi[i])
        return sample * self.scaling

    def get_series(self, npts: int) -> np.ndarray:
        """Retrieve an array of npts samples."""
        if npts > np.iinfo(int).max:
            raise ValueError("""Argument 'npts' must be an integer <= {}. If you
            want to obtain more samples, run get_series() several times 
            and concatenate the results.""".format(np.iinfo(int).max))
        samples = self.whitenoise.get_series(npts)
        for i in range(0, len(self.a)):
            samples, self.zi[i] = signal.lfilter(
                self.a[i], self.b[i], samples, zi=self.zi[i])
        return samples * self.scaling

    def _calc_filter_coeff(self, f_min: float, f_max: float) -> tuple:
        a0 = (self.fs + f_max * np.pi) / (self.fs + f_min * np.pi)
        a1 = -1. * (self.fs - f_max * np.pi) / (self.fs + f_min * np.pi)
        b1 = (self.fs - f_min * np.pi) / (self.fs + f_min * np.pi)
        return (a0, a1, b1)


class PinkNoise(AlphaNoise):
    """Pink noise generator (1/f power spectrum)."""

    def __init__(self, f_sample: float, f_min: float, f_max: float,
                 init_filter: bool = True):
        """Create a PinkNoise instance."""
        AlphaNoise.__init__(self, f_sample, f_min, f_max, 1.0, init_filter)
