#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: project3_module.py
Description: This module processes ECG data and loads, normalizes, filters, detects heartbeats, and analyzes HRV calculations. It also holds functions to visualize several of these processes.
Authors: Kristof R-M, Sawyer Hays
Used ChatGPT sparingly for plotting assistance as needed

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d


#%% Part 1: Collect and Load the Data

def load_data(infile, fs):
    '''
    Function to load the data from an input file and generate a corresponding time 
    array based on the sampling frequency

    Parameters
    ----------
    infile : string
        Name of the input file to load data from.
    fs : int
        Sampling frequency in Hz used to make the time array.

    Returns
    -------
    data : 1D array of size N, where N is the number of samples loaded
        Data corresponding to the input file, typically ECG signal data.
    data_time : 1D array of size N, where N is again the number of samples loaded
        Array for the time associated with the loaded data.

    '''
    
    # Load and normalize data
    data = np.loadtxt(infile) # convert to volts
    data = data[data != 0] # Remove zeros (typically happens from too much movement during recording)
    data = data - np.mean(data) # normalize data
    
    # Get corresponding time array
    data_time = np.arange(len(data)) / 1/fs
    
    return data, data_time

def establish_figure(figure_count, suptitle = None):
    '''
    Initializes a new figure for plotting and adds 1 to the figure count
    
    Parameters
    ----------
    figure_count : int
        The current figure number to initialize the new figure with
    suptitle : string, optional
        Overall figure title. Default title is None

    Returns
    -------
    figure_count+1 : int
        add 1 to the total figure count for later use

    '''
    plt.figure(figure_count, clear=True, figsize=(16, 9))
    plt.suptitle(suptitle)
    
    return figure_count+1
    
def plot_data(x, y, title, xlabel='Time (s)', ylabel='ECG Voltage (V)', alpha=1, label=None):
    '''
    Plots ECG data with options for xlabel, ylabel, plot label, and transparency

    Parameters
    ----------
    x : 1D array of size N, where N is typically time in seconds
        The data corresponding to the x-axis, typically time values for the ECG signal
    y : 1D array of size N, where N is typically ECG values
        The data corresponding to the y-axis, typically ECG signal values
    title : string
        Title for the plot
    xlabel : string, optional
        Label for the x-axis. The default is 'Time (s)'.
    ylabel : string, optional
        Label for the y-axis. The default is 'ECG Voltage (V)'.
    alpha : float, optional
        Level of transparency from 0 (fully transparent) to 1 (fully opaque). The default is 1 (opaque).
    label : string, optional
        Label for the line plot. The default is None.

    Returns
    -------
    None.

    '''
    
    # Plot ECG data and time
    plt.plot(x, y, alpha=alpha, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # Add legend if label is provided
    if label:
        plt.legend()
   
#%% Part 2: Filter Your Data

def create_filter(numtaps, cutoff_frequency, window, filter_type, fs):
    '''
    Creates a FIR filter based on the given parameters and returns the filter's time and frequency domain representations.

    Parameters
    ----------
    numtaps : int
        The number of taps (or coefficients) for the filter. This defines the filter's order and affects the sharpness of the cutoff.
    cutoff_frequency : float or tuple of float
        The cutoff frequency (or frequencies if it's a bandpass filter) for the filter in Hz.
    window : str or tuple of str
        The windowing function used to shape the filter. Common windows include 'hamming', 'hann', 'blackman', etc.
    filter_type : str
        The type of filter to create. Should be 'lowpass', 'highpass', 'bandpass', or 'bandstop' depending on the application.
    fs : int
        Sampling frequency in Hz used to make the time array and get the corresponding frequencies

    Returns
    -------
    h_t : 1D array of size N, where N is the number of filter taps (numtaps)
        The impulse response of the filter in the time domain
    t_filter : 1D array of size N, where N is the number of filter taps (numtaps)
        The corresponding time array for the filter's impulse response.
    H_f : 1D array of size N, where N is the number of filter taps (numtaps)
        The frequency response of the filter (transfer function).
    f_filter : 1D array of size N, where N is the number of filter taps (numtaps)
        The corresponding frequency array for the filter's frequency response.
    '''
    
    # Get filter coefficients
    h_t = signal.firwin(numtaps, cutoff_frequency, window=window, pass_zero=filter_type, fs=fs)  # Impulse response of the filter
    t_filter = np.arange(0, len(h_t)/fs, 1/fs)  # Get corresponding time for the filter
    H_f = np.fft.rfft(h_t)  # Get frequency response of filter
    f_filter = np.fft.rfftfreq(len(h_t), 1/fs)  # Get corresponding frequencies
    
    return h_t, t_filter, H_f, f_filter


def apply_filter(ecg_signal, h_t, fs):
    '''
    Applies a filter to the ECG signal using filtfilt and the given filter coefficients

    Parameters
    ----------
    ecg_signal : 1D array of size N, where N is the number of ECG samples
        The raw ECG signal to filter.
    h_t : 1D array of size N, where N is the number of taps in the filter
        The impulse response that will be applied to the ECG signal.
    fs : int
        Sampling frequency in Hz used to make the time array.


    Returns
    -------
    ecg_filtered : 1D array of size N, where N is the number of ECG data points
        The filtered ECG signal in the time domain.
    ecg_filtered_fft : 1D array of size N, where N is the size of the FFT
        Filtered in ECG in the frequency domain (FFT)
    ecg_signal_fft_frequencies : 1D array of size N, where N is the size of the FFT
        Corresponding frequencies for the ECG signal in the frequency domain
    '''
    
    # Apply the FIR filter using filtfilt
    ecg_filtered = signal.filtfilt(h_t, 1, ecg_signal)
    
    # Frequency-domain representation of the filtered signal
    ecg_filtered_fft = np.fft.rfft(ecg_filtered)
    ecg_signal_fft_frequencies = np.fft.rfftfreq(ecg_signal.size, 1/fs)
    
    return ecg_filtered, ecg_filtered_fft, ecg_signal_fft_frequencies

def butterworth_filter(data, lowcut, highcut, fs, order):
    '''
    Applies a Butterworth bandpass filter to the input data.

    Parameters
    ----------
    data : 1D array of size N, where N is the number of data points
        The input signal.
    lowcut : int
        The lower cutoff frequency of the bandpass filter.
    highcut : int
        The upper cutoff frequency of the bandpass filter.
    fs : int
        The sampling rate of the signal in Hz
    order : int
        The order of the Butterworth filter

    Returns
    -------
    filtered_data : 1D array of size N, where N is the number of data points
        The filtered data signal.

    '''

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

#%% Part 3: Detect Heartbeats

def detect_heartbeats(filtered_ecg_data, fs, refractory_period_s=200, min_peak_height=0.05):
    '''
    Detects heartbeats in a filtered ECG signal using a peak detection method.

    Parameters
    ----------
    filtered_ecg_data : 1D array of size N, where N is the number of ECG data points
        The filtered ECG signal.
    fs : int
        The sampling frequency of the signal in Hz.
    refractory_period_s : float, optional
        Refractory period in seconds to prevent double counting. The default is 200.
    min_peak_height : float, optional
        Minimum height of peaks to be considered (as a fraction of max amplitude). The default is 0.05.

    Returns
    -------
    heartbeat_times : 1D array of of size N, where N is the number of heartbeats
        Times at which a heartbeat occurred.
    filtered_peaks : 1D array of size N, where N is the number of heartbeats
        Indices of the heartbeats in the filtered ECG signal.

    '''

    # Find peaks using scipy's find_peaks
    min_height = min_peak_height * np.max(filtered_ecg_data)  # Set minimum peak height
    peaks, properties = find_peaks(filtered_ecg_data, height=min_height, distance=fs*0.5)  # Distance prevents multiple peaks from the same heartbeat
    
    # Convert the peak indices to times (in seconds)
    heartbeat_times = peaks / fs

    # Remove any heartbeats that are too close (based on the refractory period)
    refractory_period_samples = int((refractory_period_s / 1000) * fs) #in ms
    
    filtered_peaks = []
    for peak_index in range(len(peaks)):
        # Check if filtered_peaks is empty or if the current peak is too close to previous peak
        if len(filtered_peaks) == 0 or (peaks[peak_index] - filtered_peaks[-1]) > refractory_period_samples:
            filtered_peaks.append(peaks[peak_index])

    # Convert the filtered peak indices to times (in seconds)
    filtered_peaks = np.array(filtered_peaks)
    heartbeat_times = filtered_peaks / fs  # Convert the filtered indices to times in seconds

    return heartbeat_times, filtered_peaks

def plot_heartbeat_detection(file, filtered_data, data_time, fs):
    '''
    Plot the detected heartbeats on a filtered ECG signal.

    Parameters:
    -----------
        file : string
            File name for labeling the plot.
        filtered_data : 1D array of size N, where N is the number of filtered data points
            Filtered ECG signal.
        data_time (numpy.ndarray): 1D array of size N, where N is the number of filtered data points
            Time array corresponding to the ECG data.
        fs : int
            Sampling frequency in Hz.
    Returns:
    --------
        None.
    '''
    
    heartbeat_times, peak_indices = detect_heartbeats(filtered_data, fs)
    heartbeat_times, peak_indices = detect_heartbeats(filtered_data, fs)
    
    # Plot each activity in its own subplot
    plt.plot(data_time, filtered_data, label='Filtered Signal', color='black')
    plt.plot(heartbeat_times, filtered_data[peak_indices], 'ro', label='Detected Heartbeats')
    plt.title(f'Detected Heartbeats for {file[:-4]}')
    plt.xlabel('Time (s)')
    plt.ylabel('ECG Voltage (V)')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    
    
#%% Part 4: Calculate Heart Rate Variability

def interpolate_ibis(heartbeat_times, ibis, dt=0.1):
    '''
    Interpolates IBIs over a given time grid using linear interpolation.
    
    Parameters:
    -----------
    heartbeat_times : 1D numpy array of size N, where N is the number of heartbeats
        Array of heartbeat times in seconds.
    
    ibis : 1D numpy array of size N, where N is the number of IBIs
        Array of Inter-Beat Intervals (IBIs) in milliseconds.
    
    dt : float
        Spacing of intervals for interpolation, in seconds.

    Returns:
    --------
    interpolated_ibis : 1D numpy array of size N, where N is the number of interpolated points
        Array of interpolated IBIs
'''
    
    # Create time points for interpolation
    start_time = heartbeat_times[0]
    end_time = heartbeat_times[-2] + (heartbeat_times[-1] - heartbeat_times[-2]) / 2
    unknown_times = np.arange(start_time, end_time, dt)
    
    # Perform the interpolation
    ibi_interpolation = interp1d(heartbeat_times[:-1], ibis, kind='linear', fill_value='extrapolate')
    interpolated_ibis = ibi_interpolation(unknown_times)
    
    return interpolated_ibis

#%% Part 5: Get HRV Frequency Band Power

def convert_to_frequency(ecg_data, fs):
    '''
    Computes the frequency spectrum of ECG data.

    Parameters:
    -----------
        ecg_data : 1D array of size N, where N is the number of data points
            Input ECG signal.
        fs : int
            Sampling frequency in Hz.

    Returns:
    --------
        positive_frequencies : 1D array of size N, where N is the number of positive frequencies
            Positive frequency values.
        magnitude_db : 1D array of size N, where N is the number of positive frequency values
            Magnitude of the frequency spectrum in dB
    '''
    
    fft_result = np.fft.fft(ecg_data)
    fft_freq = np.fft.fftfreq(len(ecg_data), d=1/fs)

    #get x axis and set limits, used chat gpt to debug this part
    positive_frequencies = fft_freq[np.logical_and(fft_freq >= 0, fft_freq <= 0.5)]
    positive_frequencies = positive_frequencies[1:]
    #get y axis
    positive_fft = np.abs(fft_result[np.logical_and(fft_freq >= 0, fft_freq <= 0.5)])[1:]
    
    magnitude_db = 20 * np.log10(positive_fft + 1e-10)
    
    return positive_frequencies, magnitude_db

def get_lf_hf_ratio(infile, filtered_ecg_dict, fs, band1_lf = 0.04, band2_lf = 0.15, band1_hf = 0.15, band2_hf = 0.4):
    '''
    Calculates the LF/HF ratio from frequency bands.

    Parameters:
    -----------
        infile : str
            File name for labeling.
        filtered_ecg_dict : dict
            Dictionary containing filtered ECG data.
        fs : int
            Sampling frequency in Hz.
        band1_lf : float, optional
            Lower limit of LF band in Hz. Default is 0.04.
        band2_lf : float, optional
            Upper limit of LF band in Hz. Default is 0.15.
        band1_hf : float, optional
            Lower limit of HF band in Hz. Default is 0.15.
        band2_hf : float, optional
            Upper limit of HF band in Hz. Default is 0.4.

    Returns:
    --------
        lf_hf_ratio : float
            LF/HF power ratio.
        frequency_range : 1D array of size N, where N is the number of frequency points
            Frequency range for plotting.
        magnitude_power : 1D array of size N, where N is the number of frequency points
            Power spectrum.
        band1_indices : 1D array of size M, where M is the number of indices in the LF band
            Indices of the LF band.
        band2_indices : 1D array of size M, where M is the number of indices in the HF band
            Indices of the HF band.
    '''
    
    # Extract ECG data from dictionary input
    ecg_data, ecg_filtered, ecg_time = filtered_ecg_dict[infile]
    
    # Compute the frequency spectrum
    frequency_range, magnitude_db = convert_to_frequency(ecg_data, fs)
    magnitude_power = 10**(magnitude_db / 10)

    # Find indices for each band
    band1_indices = np.where((frequency_range >= band1_lf) & (frequency_range <= band2_lf))
    band2_indices = np.where((frequency_range >= band1_hf) & (frequency_range <= band2_hf))

    # Calculate mean power for each band
    lf_mean_power = np.mean(magnitude_power[band1_indices])
    hf_mean_power = np.mean(magnitude_power[band2_indices])

    # Calculate the LF/HF ratio
    lf_hf_ratio = lf_mean_power / hf_mean_power
    
    return lf_hf_ratio, frequency_range, magnitude_power, band1_indices, band2_indices

def plot_frequency_spectrum(frequency_range, magnitude_power, band1_indices, band2_indices, infile, color='blue', label='Frequency Spectrum'):
    '''
    Plots the frequency spectrum with labeled LF and HF bands.

    Parameters:
    -----------
        frequency_range : 1D array of size N, where N is the number of frequency points
            Frequency range for the spectrum.
        magnitude_power : 1D array of size N, where N is the number of frequency points
            Power spectrum.
        band1_indices : 1D array of size M, where M is the number of indices in the LF band
            Indices of the LF band.
        band2_indices : 1D array of size M, where M is the number of indices in the HF band
            Indices of the HF band.
        infile : str
            File name for labeling.
        color : str, optional
            Line color for the plot. Default is 'blue'.
        label : str, optional
            Label for the frequency spectrum. Default is 'Frequency Spectrum'.

    Returns:
    --------
        None.
    '''
    
    # Plot the frequency spectrum in the current subplot with labeled LF and HF bands
    plt.plot(frequency_range, magnitude_power, label=label, color=color)
    plt.plot(frequency_range[band1_indices], magnitude_power[band1_indices], color='red', label='LF Band (0.04-0.15 Hz)')
    plt.plot(frequency_range[band2_indices], magnitude_power[band2_indices], color='green', label='HF Band (0.15-0.4 Hz)')
    plt.title(f"FFT Spectrum for {infile[:-4]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Power)")
    plt.grid()
    plt.legend()











