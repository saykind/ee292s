import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load data
filename = 'signal200.npy'
data = np.load(filename)
N = 20000 # number of samples
f0 = 100.12 # Hz
dt = 1/f0 # s
t = np.arange(0, N*dt, dt) # time

# Find peaks
peaks, _ = signal.find_peaks(data, height=0.03, threshold=None, distance=30)
peak_diff = np.diff(peaks)

# max, avg, std HR spacing
HR_spacing = np.max(peak_diff) / f0
HR_std = np.std(peak_diff) / f0
HR_avg = np.mean(peak_diff) / f0
print(f"Max HR spacing: {HR_spacing:.3f}s, avg: {HR_avg:.3f} std: {HR_std:.3f}s")

# Plot
plt.plot(t, data, '.-')
plt.show()