import numpy as np
import matplotlib.pyplot as plt
import scipy
import time


def make_filename(N=20000, RATE=100):
    """ Makes a filename for saving data."""
    t = time.localtime()
    filename = f"data{RATE}Hz/" \
               f"data{N // RATE}_{t.tm_year}-{t.tm_mon}-{t.tm_mday}_" \
               f"{t.tm_hour}-{t.tm_min}-{t.tm_sec}.npz"
    return filename


def repack(signal_filename, freqs_filename, new_filename='data.npz'):
    """ Load np.array from signal_filename and freqs_filename and save them
    in a signgle npz file.
    """
    signal = np.load(signal_filename)
    freqs = np.load(freqs_filename)
    np.savez(new_filename, signal=signal, freqs=freqs)


def interpolate(signal, freqs, dt=None):
    """ Interpolate signal to have a sampling interval of dt."""
    assert len(signal) == len(freqs)
    dts = 1 / freqs
    if dt is None:
        dt = np.mean(dts)
    naive_ts = np.linspace(0, len(signal) * dt, len(signal))
    old_ts = np.cumsum(dts) - dts
    N = old_ts[-1] // dt
    new_ts = np.arange(0, N * dt, dt)
    new_signal = np.interp(new_ts, old_ts, signal)
    return new_signal, new_ts, old_ts, naive_ts


def load(filename):
    """ Load data from filename."""
    data = np.load(filename)
    signal = data['signal']
    freqs = data['freqs']
    s, t, _, _ = interpolate(signal, freqs)
    return s, t


def fourier(signal, ts):
    """ Compute fourier transform of signal."""
    N = len(signal)
    signal_fft = np.fft.fft(signal)
    dt = ts[1] - ts[0]
    fs = np.linspace(0, 1 / dt, N)
    return np.abs(signal_fft), fs


def find_peaks(signal, ts, min_height=0.015, min_dist=0.6):
    """ Find indexes of peaks in signal.
    Args:
        signal: np.array
        ts: np.array
        min_height: float
            Minimum height of peaks.
        min_dist: float
            Minimum distance between peaks in seconds.
    Returns:
        peaks: np.array
            Indexes of peaks.
        peak_ts: np.array
            Timestamps of peaks.
    """

    dt = ts[1] - ts[0]
    peaks, _ = scipy.signal.find_peaks(signal, height=min_height, distance=min_dist / dt)

    # max, avg, std HR spacing
    peak_ts = ts[peaks]
    peak_diff = np.diff(peak_ts)
    print(f" Avg spacing: {np.mean(peak_diff):.3f} sec\n Standard deviation: {np.std(peak_diff):.3f} sec")
    print(f" Max spacing: {np.max(peak_diff):.3f} sec\n Min spacing: {np.min(peak_diff):.3f} sec")
    return peaks, peak_ts