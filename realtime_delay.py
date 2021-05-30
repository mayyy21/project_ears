"""
Listens for impulsive sounds. When one is heard, it displays the recording of
the left and right channels on the top, and their cross-correlation on the
bottom, finds the peak using parabolic/quadratic interpolation, plots
the peak as a red dot and prints the measured distance between the microphones.

If you disable the crest factor check, and play white noise with a phone, you
can see the red dot move back and forth as you vary the angle to the
microphones.
"""

from __future__ import division, print_function
import numpy as np
from matplotlib.mlab import rms_flat
from matplotlib import pyplot as plt
import pyaudio
from scipy.signal import fftconvolve, butter, lfilter
from PyQt4.QtGui import QApplication
from parabolic import parabolic


fs = 96000  # sampling rate
format = pyaudio.paFloat32  # max = 1.0
channels = 2
chunk = int(fs/4)
c = 343  # m/s = speed of sound


def crest_factor(signal):
    """
    Crest factor of a 1D signal
    """
    peak = np.amax(np.absolute(signal))
    rms = rms_flat(signal)
    return peak / rms


def callback(in_data, frame_count, time_info, status):
    """
    Called on each incoming frame to process data
    """
    global result
    global result_waiting

    if in_data:
        print('.', end='')
        result = np.fromstring(in_data, dtype=np.float32)
        result = np.reshape(result, (chunk, 2))  # stereo
        result_waiting = True
    else:
        print('no input')

    return None, pyaudio.paContinue


# Initialize blank plots
plt.figure(1)
plt.subplot(2, 1, 1)
t = np.arange(chunk)/fs
plt_L = plt.plot(t,  np.ones(chunk), 'blue')[0]
plt_R = plt.plot(t, -np.ones(chunk), 'red')[0]  # Red = Right
plt.margins(0, 0.1)
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.title('Recording')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
lags = np.arange(chunk)-chunk/2
plt_corr = plt.plot(lags, np.zeros(chunk))[0]
plt_peak = plt.plot(0, 0, 'ro')[0]
plt.margins(0, 0.1)
plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
plt.title('Cross-correlation %.2f' % 0)
plt.xlabel('Lag [samples]')
plt.ylabel('Amplitude')
plt.margins(0.1, 0.1)
plt.xlim(-100, 100)
plt.ylim(-10, 10)

# Generate quick high-pass filter for motor hums, etc
cutoff = 500  # Hz
B, A = butter(1, cutoff / (fs/2), btype='high')

# Initialize global variables used by callback function
result = np.zeros((chunk, channels))
result_waiting = False

p = pyaudio.PyAudio()

if not p.is_format_supported(fs, 0, channels, format):
    p.terminate()
    raise RuntimeError('Format not supported')

# open stream using callback (3)
stream = p.open(format=format,
                channels=channels,
                rate=fs,
                output=False,
                input=True,
                frames_per_buffer=chunk,
                stream_callback=callback)

print('Press Ctrl+C or close plot window to stop')

try:
    stream.start_stream()

    try:
        while plt.fignum_exists(1):  # user has not closed plot
            if result_waiting:
                result_waiting = False

                # High-pass filter
                result = lfilter(B, A, result, axis=0)
                sig_L = result[:, 0]
                sig_R = result[:, 1]

                # Only update plots on impulsive sound
                # (Disable this for continuous tracking of continuous sources,
                # like a phone playing white noise)
                cf = crest_factor(sig_L)
                if cf > 18:
                    plt_L.set_data(t, sig_L)
                    plt_R.set_data(t, sig_R)

                    corr = fftconvolve(sig_R, sig_L[::-1], mode='same')

                    # Update plots
                    plt.subplot(2, 1, 1)
                    plt.title('Recording (crest factor: {:.2f})'.format(cf))

                    plt.subplot(2, 1, 2)
                    plt_corr.set_data(lags, corr)
                    argpeak, amppeak = parabolic(corr, np.argmax(corr))
                    plt_peak.set_data(argpeak-chunk/2, amppeak)
                    plt.ylim(np.amin(corr)*1.1, np.amax(corr)*1.1)

                    distance = (argpeak-chunk/2) / fs * c  # m
                    plt.title('Cross-correlation '
                              '(dist: {:.2f} cm)'.format(distance * 100))
                    plt.draw()

                # doesn't work in Spyder without this
                # https://code.google.com/p/spyderlib/issues/detail?id=459
                QApplication.processEvents()

    except KeyboardInterrupt:
        print('\nCtrl+C: Quitting')
    else:
        print('\nFigure closed: Quitting')

finally:
    plt.close('all')

    stream.stop_stream()
    stream.close()

    p.terminate()
