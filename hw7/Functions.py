import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.io as sio
from pylab import *
import os

def readPoiSpikes(fileName, Fs):

    binWidth = 1 / (Fs)
    try:
        spikeData = sio.loadmat(fileName)
        spikes = spikeData['spikes']
    except KeyError:
        print("Error: file not found.")
        return None
    spikeTrain = np.squeeze(spikes)

    # Check if the loaded spike train is empty
    if len(spikes) == 0:
        print("Error: Empty spike train dataset.")
        return None, None

    # Convert spike train to column vector if it's a row vector
    if spikes.shape[0] == 1:
        spikeTrain = np.reshape(spikes, (-1, 1))

    # Check if the spike train contains any data
    if spikes.shape[0] == 0:
        print("Error: Empty spike train dataset.")
        return None, None

    # Convert spike times to set binwidth
    maxTime = int(np.ceil(spikeTrain[-1]))
    minTime = int(np.ceil(spikeTrain[0]))
    T = maxTime - minTime
    numBins = int(np.ceil(T / binWidth))
    spikeTrain_binned, timeBins = np.histogram(spikeTrain, bins=numBins)
    return spikeTrain_binned

def generatePoiSpikes(r, dt, totalSize):   #refractory peride?
    M = int(np.floor(totalSize / dt))
    prob_of_spike = r*dt
    boolean_spike_train = np.random.rand(M) < prob_of_spike
    spikeTrain= boolean_spike_train*1
    return spikeTrain


def calc_isi(binaryTrain):
    spikes_indices, = np.nonzero(binaryTrain)
    num_spikes = len(spikes_indices)  #FOR NP.NAH
    isi = np.diff(spikes_indices)
    return isi

def calcCV(spikeTrain):
    isi = calc_isi(spikeTrain)
    CV = (np.std(isi))/(np.mean(isi))
    return CV




def calcFF(spikeTrain):
    FF = (np.var(spikeTrain))/ (np.mean(spikeTrain))
    return FF


def calcRate(spikeTrain, window, dt):
    if window == 0:
        num_spikes = np.sum(spikeTrain)
        totalTime = len(spikeTrain) * dt
        rateOfFire = num_spikes / totalTime
        return rateOfFire

    else:
        num_bins = len(spikeTrain)
        num_windows = int(num_bins / window)
        remainder = num_bins % window

        if remainder > 0:
            num_windows += 1

        rateOfFire = np.zeros(num_windows)
        for i in range(num_windows):
            start_idx = i * window
            end_idx = min((i + 1) * window, num_bins)
            num_spikes = np.sum(spikeTrain[start_idx:end_idx])
            time_window = (end_idx - start_idx) * dt
            rateOfFire[i] = num_spikes / time_window

        return rateOfFire

# Example usage:
# spikeTrain = array of spike timings
# window = length of the window for calculating firing rate
# dt = time step for discretization
# rateOfFire = calcRate(spikeTrain, window, dt)


