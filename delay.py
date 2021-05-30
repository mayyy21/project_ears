"""
Measure the distance between two microphones based on the delay of a
clap from a point that is collinear with the two microphones.

Make sure your signal isn't clipping.

First test:
2 electrets 26 cm apart, hand clap
Distance:            -25.72 cm
Sub-sample distance: -26.02 cm

Second test:
Laptop's stereo mics, 6.9 cm apart, snapping fingers to one side
Distance:            6.79 cm
Sub-sample distance: 6.85 cm
"""

from __future__ import division
import soundfile as sf
from scipy.signal import butter, lfilter, fftconvolve
from numpy import argmax

# download from https://gist.github.com/endolith/255291#file-parabolic-py
from parabolic import parabolic

c = 343  # m/s = speed of sound

signal, fs = sf.read('stereo recording.flac')

# Quick high-pass filter
cutoff = 500  # Hz
B, A = butter(1, cutoff / (fs/2), btype='high')
signal = lfilter(B, A, signal, axis=0)

# Cross-correlation of the two channels (same as convolution with one reversed)
corr = fftconvolve(signal[:, 0], signal[::-1, 1], mode='same')

# Find the offset of the peak. Zero delay would produce a peak at the midpoint
delay = int(len(corr)/2) - argmax(corr)
distance = delay / fs * c
print("Distance: %.2f cm" % (distance * 100))

# More accurate sub-sample estimation with parabolic interpolation:
delay = int(len(corr)/2) - parabolic(corr, argmax(corr))[0]
distance = delay / fs * c
print("Sub-sample distance: %.2f cm" % (distance * 100))
