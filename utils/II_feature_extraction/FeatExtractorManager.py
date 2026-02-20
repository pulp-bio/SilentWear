# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Feature extraction functions for EMG signals.

This module contains classes to compute time-domain, frequency-domain,
and wavelet-based features from EMG signal windows.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import welch
import pywt


class FeatureRegistry:
    """Central registry of all supported EMG features."""

    TIME_DOMAIN = [
        "rms",
        "max",
        "min",
        "std",
        "var",
        "q25",
        "q75",
        "mean",
        "zcr",
    ]

    FREQUENCY_DOMAIN = [
        "mean_f",
        "peak_f",
        "tot_p",
        "mean_p",
        "spec2",
        "spec3",
        "ratio_f",
    ]

    WAVELET_DOMAIN = [
        "c3Am",
        "c3As",
        "cD3m",
        "cD3s",
        "cD2m",
        "cD2s",
        "cD1m",
        "cD1s",
    ]

    # Flat list of all features
    ALL = TIME_DOMAIN + FREQUENCY_DOMAIN + WAVELET_DOMAIN

    @classmethod
    def validate(cls, features: list[str]) -> None:
        """Raise error if any feature is unsupported."""
        invalid = [f for f in features if f not in cls.ALL]
        if invalid:
            raise ValueError(f"Invalid feature(s): {invalid}. Allowed features: {cls.ALL}")


class TimeDomainFeatureExtractor:
    """Extracts time-domain features from EMG signals."""

    @staticmethod
    def extract(window_data: np.ndarray) -> Dict[str, float]:
        """Compute time-domain features from a signal window.

        Args:
            window_data: 1D array of signal values for one channel.

        Returns:
            Dictionary with time-domain feature values.
        """
        sign_changes = np.where(np.diff(np.sign(window_data)))[0]

        return {
            "rms": float(np.sqrt(np.mean(window_data**2))),
            "max": float(np.max(window_data)),
            "min": float(np.min(window_data)),
            "std": float(np.std(window_data)),
            "var": float(np.var(window_data)),
            "q25": float(np.quantile(window_data, 0.25)),
            "q75": float(np.quantile(window_data, 0.75)),
            "mean": float(np.mean(window_data)),
            "zcr": int(len(sign_changes)),
        }


class FrequencyDomainFeatureExtractor:
    """Extracts frequency-domain features from EMG signals."""

    DEFAULT_NPERSEG = 100

    @classmethod
    def extract(
        cls,
        window_data: np.ndarray,
        fs: float,
        nperseg: int = DEFAULT_NPERSEG,
        low_freq_band: Tuple[int, int] = (0, 125),
        high_freq_band: Tuple[int, int] = (125, 250),
    ) -> Dict[str, float]:
        """Compute frequency-domain features from a signal window.

        Args:
            window_data: 1D array of signal values for one channel.
            fs: Sampling frequency in Hz.
            nperseg: Length of each segment for Welch's method.
            low_freq_band: Low frequency band tuple (start, end).
            high_freq_band: High frequency band tuple (start, end).

        Returns:
            Dictionary with frequency-domain feature values.
        """
        f, Pxx = welch(window_data, fs, nperseg=nperseg)
        total_power = np.sum(Pxx)

        low_freq_power = cls._calculate_band_power(f, Pxx, low_freq_band)
        high_freq_power = cls._calculate_band_power(f, Pxx, high_freq_band)
        freq_ratio = low_freq_power / high_freq_power if high_freq_power != 0 else 0

        return {
            "mean_f": float(np.sum(f * Pxx) / total_power) if total_power > 0 else 0.0,
            "peak_f": float(f[np.argmax(Pxx)]),
            "tot_p": float(total_power),
            "mean_p": float(np.mean(Pxx)),
            "spec2": float(np.sum(f * f * Pxx)),
            "spec3": float(np.sum(f * f * f * Pxx)),
            "ratio_f": float(freq_ratio),
        }

    @staticmethod
    def _calculate_band_power(
        frequencies: np.ndarray, power: np.ndarray, band: Tuple[int, int]
    ) -> float:
        """Calculate power within a frequency band.

        Args:
            frequencies: Frequency array from Welch's method.
            power: Power spectral density array.
            band: Tuple of (low_freq, high_freq) defining the band.

        Returns:
            Total power within the specified frequency band.
        """
        indices = np.where((frequencies >= band[0]) & (frequencies <= band[1]))[0]
        return float(np.sum(power[indices]))


class WaveletFeatureExtractor:
    """Extracts wavelet-based features from EMG signals."""

    DEFAULT_WAVELET = "db4"
    DEFAULT_LEVEL = 3

    @classmethod
    def extract(
        cls,
        window_data: np.ndarray,
        wavelet: str = DEFAULT_WAVELET,
        level: int = DEFAULT_LEVEL,
    ) -> Dict[str, float]:
        """Compute wavelet-based features from a signal window.

        Uses discrete wavelet transform to decompose the signal and extract
        statistical features from approximation and detail coefficients.

        Args:
            window_data: 1D array of signal values for one channel.
            wavelet: Wavelet type (default: 'db4').
            level: Decomposition level (default: 3).

        Returns:
            Dictionary with wavelet feature values.
        """
        coeffs = pywt.wavedec(window_data, wavelet, level=level)

        c3A = coeffs[0]
        cD3, cD2, cD1 = coeffs[1:]

        return {
            "c3Am": float(np.mean(c3A)),
            "c3As": float(np.std(c3A)),
            "cD3m": float(np.mean(cD3)),
            "cD3s": float(np.std(cD3)),
            "cD2m": float(np.mean(cD2)),
            "cD2s": float(np.std(cD2)),
            "cD1m": float(np.mean(cD1)),
            "cD1s": float(np.std(cD1)),
        }


class FeatureExtractor:
    """Main feature extraction pipeline combining all feature types."""

    def __init__(
        self,
        fs: Optional[float] = None,
        nperseg: int = 100,
        low_freq_band: Optional[Tuple[int, int]] = None,
        high_freq_band: Optional[Tuple[int, int]] = None,
        # config: Optional[EMGConfig] = None,
    ):
        """Initialize feature extractor.

        Args:
            fs: Sampling frequency in Hz (optional if config provided).
            nperseg: Length of each segment for Welch's method.
            low_freq_band: Low frequency band tuple (start, end).
            high_freq_band: High frequency band tuple (start, end).
            config: Optional EMGConfig object for configuration.
        """
        # if config is not None:
        #     self.fs = config.emg_fs
        #     self.low_freq_band = config.low_freq_band
        #     self.high_freq_band = config.high_freq_band

        if fs is None:
            raise ValueError("Either fs or config must be provided")
        self.fs = fs
        self.low_freq_band = low_freq_band or (0, 125)
        self.high_freq_band = high_freq_band or (125, 250)

        self.nperseg = nperseg
        self.time_extractor = TimeDomainFeatureExtractor()
        self.freq_extractor = FrequencyDomainFeatureExtractor()
        self.wavelet_extractor = WaveletFeatureExtractor()

    def extract_window_features(self, window_data: np.ndarray) -> Dict[str, float]:
        """Extract all features from a single signal window.

        Combines time-domain, frequency-domain, and wavelet features.

        Args:
            window_data: 1D array of signal values for one channel.

        Returns:
            Dictionary containing all extracted features.
        """
        features = {}
        features.update(self.time_extractor.extract(window_data))
        features.update(
            self.freq_extractor.extract(
                window_data,
                self.fs,
                self.nperseg,
                self.low_freq_band,
                self.high_freq_band,
            )
        )
        features.update(self.wavelet_extractor.extract(window_data))
        return features

    @staticmethod
    def _build_feature_name(feature_name: str, window_num: int, channel_tag: str) -> str:
        """Build feature name with window number and channel tag.

        Args:
            feature_name: Base feature name.
            window_num: Window number.
            channel_tag: Channel tag (e.g., '01n').

        Returns:
            Formatted feature name.
        """
        return f"{feature_name}_{window_num}_{channel_tag}"
