# Import dependencies
import data as data
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.io import loadmat
import os
from scipy.signal import spectrogram
from scipy.io import loadmat

data = loadmat('EEG-Signal-Homework.mat')  # Load the EEG data
EEG = data['EEG'][:, 0]                 # Extract the EEG variable
t = data['t'][0]                        # Extract the time variable

plot(t, EEG)                            # Plot the data in the time domain
xlabel('Time [s]')                      # Label the time axis
ylabel('Voltage [$\mu V$]')             # Label the voltage axis
autoscale(tight=True)                   # Minimize white space
show()

# plot(t[:25], EEG[:25], 'o-')    # Plot the first 25 points
# xlabel('Time [s]')              # Label the time axis
# ylabel('Voltage [$\mu V$]')     # Label the voltage axis
# show()
print(t[1] - t[0])
sampling_rate = 1000
dt =t[1] - t[0]
N = len(EEG) # Your code here | Hint: "EEG" is an array where each index in the array is a sample
T = N * dt
f_s = 1/dt

EEG_transformed = np.fft.rfft(EEG)
''' Hint: look up the fourier transform functions in Numpy, and use the Discrete Fourier Transform for real input function'''

spectrum = (2 * dt ** 2 / T * EEG_transformed  * EEG_transformed.conj()).real
faxis = arange(len(spectrum)) / T.max()
plot(faxis, spectrum)
xlim([0, 100]) # Setting the frequency range.
xlabel('Frequency (Hz)')
ylabel('Power [$\mu V^2$/Hz]')
show()

plot(faxis, 10 * log10(spectrum / max(spectrum)))  # Plot the spectrum in decibels.
xlim([0, 100])                           # Setting the frequency range.
ylim([-60, 0])                           # Setting the decibel range.
xlabel('Frequency [Hz]')
ylabel('Power [dB]')
show()

decibel_value_at_60 = 0


f, t, Sxx = spectrogram(
    EEG,
    fs=f_s,
    nperseg=int(f_s),
    noverlap=int(f_s * 0.95))
pcolormesh(t, f, 10 * log10(Sxx), cmap='jet')
colorbar()
ylim([0, 70])
xlabel('Time [s]')
ylabel('Frequency [Hz]')
show()