#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:11:43 2024
Filename: project3_script.py

Description: This script utilizes project3_module to load ECG data from 3 different activities, \
    filter it using a finite and infinite bandpass filter, detect heartbeats, calculate\
        heart rate variability, and calculate HRV frequency band power. 

Authors: Kristof R-M, Sawyer Hays
Used ChatGPT for plotting assistance as necessary

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import project3_module as p3m


#%% Part 1: Collect and Load Data

# Define the sampling frequency and an array of all the infiles
fs = 500 # Hz
figure_count = 1 # Which figure it's on

# Initiate empty dictionary to store all relevant ecg data
ecg_signal_data = {}

# Define infiles with array containing all of them
infiles = ['Rest.txt', 'Relax.txt', 'PhysicalStress.txt', 'MentalStress.txt']

# Establish figure
figure_count = p3m.establish_figure(figure_count, 'Raw ECG Signal for All Activities')

# Calculate rows and columns for subplots
col_count = 2  # Two columns per row
row_count = (len(infiles) + 1) // col_count  # The number of rows needed (rounding up)

# Initialize concatenated data signal
concatenated_ecg_data = np.empty(0)

# Initialize raw ECG dictionary
raw_ecg_dict = {}

# Loop through the infile arrays and call the module functions to get the ecg data and times and plot it
for plot_count, infile in enumerate(infiles):
    # Get ecg data and time, establish subplot, and plot ecg data
    ecg_data, ecg_time = p3m.load_data(infile, fs)
    ecg_data *= 5 / 1023 # scale the ECG data to volts
    # Append the data to the dictionary for later
    raw_ecg_dict[infile] = (ecg_data, ecg_time)
    
    # Plot the raw ecg data
    plt.subplot(row_count, col_count, plot_count+1)
    p3m.plot_data(ecg_time, ecg_data, f'{infile[:-4]} Data')
    plt.xlim(70,75) # show 5 seconds of data
    
    # Concatenate given ecg data to the overall concatenated ecg signal
    concatenated_ecg_data = np.concatenate((concatenated_ecg_data, ecg_data))

# Plot the concatenated ECG signals
figure_count = p3m.establish_figure(figure_count)
concatenated_time = np.arange(len(concatenated_ecg_data)) / 1/fs  # corresponding concatenated time
p3m.plot_data(concatenated_time, concatenated_ecg_data, 'Concatenated ECG Data')

# Add vertical lines and labels (used ChatGPT)
start_idx = 0
for infile in infiles:
    ecg_data, _ = raw_ecg_dict[infile]
    end_idx = start_idx + len(ecg_data)
    midpoint_time = (start_idx + end_idx) // 2 / fs
    
    # Add vertical line
    if start_idx != 0:
        plt.axvline(x=start_idx / fs, color='k', linestyle='--')
        
    # Add caption
    plt.annotate(
        f'{infile[:-4]}', 
        xy=(midpoint_time, np.max(concatenated_ecg_data) * 0.9), 
        xytext=(0, 10), 
        textcoords='offset points', 
        ha='center', fontsize=10, color='k', 
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    start_idx = end_idx



#%% Part 2: Filter Your Data
 
# FIR window filtering
numtaps = 501
cutoff_frequency = [15, 50]
window = 'hamming'
filter_type = 'bandpass'

# Get filter
h_t, t_filter, H_f, f_filter = p3m.create_filter(numtaps, cutoff_frequency, window, filter_type, fs)

# Establish figure
figure_count = p3m.establish_figure(figure_count, 'Filter Time and Frequency Domain Representation')

# Plot impulse response
plt.subplot(1,2,1)
p3m.plot_data(t_filter, h_t, 'FIR Filter Impulse Response', ylabel = 'Gain')

# Plot frequency response
plt.subplot(1,2,2)
p3m.plot_data(f_filter, np.abs(H_f), 'FIR Filter Transfer Function', xlabel = 'Frequency (Hz)', ylabel = 'Gain')

# Create empty dictionary to hold filtered data
filtered_ecg_dict = {}

# Create loop to filter all data files so we can access them later
for infile in infiles:
    # Extract the ecg data from the previous dictionary
    ecg_data, ecg_time = raw_ecg_dict[infile]
    # Apply filter to the ecg data (also get filtered FFT and frequencies if wanted)
    ecg_filtered, ecg_filtered_fft, ecg_signal_fft_frequencies = p3m.apply_filter(ecg_data, h_t, fs)
    # Store the filtered data in the dictionary
    filtered_ecg_dict[infile] = (ecg_data, ecg_filtered, ecg_time)

# Extract the rest data from the dictionary
rest_ecg, rest_ecg_filtered, rest_time = filtered_ecg_dict[infiles[0]]

# Establish figure for comparing FIR and IIR filter on the rest data
figure_count = p3m.establish_figure(figure_count, 'Comparing FIR and IIR Filter Types for ECG Signals')

# Plot the FIR window original and filtered rest ECG signals
plt.subplot(1,2,1)
p3m.plot_data(rest_time, rest_ecg, 'FIR Rest ECG Original and Filtered', label='Original Rest ECG')
p3m.plot_data(rest_time, rest_ecg_filtered, 'FIR Rest ECG Original and Filtered', alpha = 0.8, label='Filtered Rest ECG')
plt.xlim(40, 50)

# Butterworth IIR filtering for comparison

# Example for resting ECG data
# Define filter parameters
lowcut = 15
highcut = 50
order = 5
rest_ecg_data, rest_time = raw_ecg_dict[infiles[0]]

# Apply filter to the ecg data (also get filtered FFT and frequencies if wanted)
rest_ecg_filtered = p3m.butterworth_filter(rest_ecg_data, lowcut, highcut, fs, order)

# Plot the IIR butterworth original and filtered rest ECG signals
plt.subplot(1,2,2)
p3m.plot_data(rest_time, rest_ecg_data, 'IIR Rest ECG Original and Filtered', label='Original Rest ECG')
p3m.plot_data(rest_time, rest_ecg_filtered, 'IIR Rest ECG Original and Filtered', alpha = 0.8, label='Filtered Rest ECG')
plt.xlim(40, 50)


#%% Part 3 Detect Heartbeats
#used chat gpt to simplify the code we wrote and put it in a loop

# Establish figure for heartbeat detection
figure_count = p3m.establish_figure(figure_count, 'Heartbeat Detection for Each Activity')

# Calculate rows and cols
col_count = 2  # Two columns per row
row_count = (len(infiles) + 1) // col_count  # The number of rows needed (rounding up)

# Collect heartbeats for all files
heartbeat_results = {}

# Loop through files and store heartbeat data
for infile in infiles:
    data, filtered_data, data_time = filtered_ecg_dict[infile]  
    heartbeat_times, peak_indices = p3m.detect_heartbeats(filtered_data, fs)
    heartbeat_results[infile] = {'heartbeat_times': heartbeat_times, 'peak_indices': peak_indices}

# File names and data
file_info = [
    (infiles[0], filtered_ecg_dict[infiles[0]]),  # Rest
    (infiles[1], filtered_ecg_dict[infiles[1]]),  # Relax
    (infiles[2], filtered_ecg_dict[infiles[2]]),  # Physical Stress
    (infiles[3], filtered_ecg_dict[infiles[3]]),  # Mental Stress
]

# Plot each file in a subplot
for plot_count, (file, (data, filtered_data, data_time)) in enumerate(file_info):
    plt.subplot(row_count, col_count, plot_count + 1)
    p3m.plot_heartbeat_detection(file, filtered_data, data_time, fs)
    plt.xlim(70, 90)
    
    # Extract heartbeat times for BPM calculation
    heartbeat_times = heartbeat_results[file]['heartbeat_times']
    
    # Calculate total time in minutes (~5 min)
    total_time_seconds = heartbeat_times[-1] - heartbeat_times[0]
    total_time_minutes = total_time_seconds / 60
    
    # Calculate the number of beats per minute (BPM)
    bpm = len(heartbeat_times) / total_time_minutes
    
    # Add annotation with BPM info on the subplot to ensure detection is logical (from ChatGPT)
    plt.annotate(f'BPM: {bpm:.2f}', 
                 xy=(0.02, 0.93),  # Position the annotation (adjust if necessary)
                 xycoords='axes fraction',  # Use axes fraction for positioning
                 fontsize=10, 
                 color='black', 
                 weight='bold',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

#%% Part 4: Calculate Heart Rate Variability

# Initialize a dictionary to store IBI data for each file
hrv_results = {}

# Loop through all 4 activities to get IBIs
for infile in infiles:
    # Extract heartbeat times
    heartbeat_times = heartbeat_results[infile]['heartbeat_times']
    
    # Calculate IBIs
    ibis = np.diff(heartbeat_times) * 1000  # Convert to milliseconds
    ibi_interpolation = interp1d(heartbeat_times[:-1], ibis, kind='linear', fill_value='extrapolate')
    interpolated_ibis = p3m.interpolate_ibis(heartbeat_times, ibis, 1/fs)
    
    # Get and store the standard deviation of the IBIs
    sdnn = np.std(interpolated_ibis)
    hrv_results[infile] = sdnn

# Extract HRV data  for each activity
activities = ['Rest', 'Relax', 'PhysicalStress', 'MentalStress']  # Define the activity labels
hrv_values = [hrv_results[infile] for infile in infiles]

# Establish figure for HRV bar plot
figure_count = p3m.establish_figure(figure_count, 'Heart Rate Variability (HRV) for Different Activities')

# Plot HRVs
plt.bar(activities, hrv_values, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Activity')
plt.ylabel('HRV (SDNN in ms)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
    
#%% Part 5: Get HRV Frequency Band Power

# Initialize lists to store LF/HF ratios and labels
lf_hf_ratios = []

# Determine grid size for subplots
num_trials = len(infiles)
rows = (num_trials + 1) // 2  # Number of rows in the 2x2 grid
cols = 2  # Two columns

# Establish figure for frequency spectrum subplots
figure_count = p3m.establish_figure(figure_count, 'Frequency Spectrums for All Activities')
    
# Process each trial and compute LF/HF ratio
for plot_count, infile in enumerate(infiles):
    # Get and append LF/HF ratio data
    lf_hf_ratio, frequency_range, magnitude_power, band1_indices, band2_indices = p3m.get_lf_hf_ratio(infile, filtered_ecg_dict, fs)
    lf_hf_ratios.append(lf_hf_ratio)
    
    # Plot the frequency spectrum with labeled LF and HF bands
    plt.subplot(rows, cols, plot_count + 1)
    p3m.plot_frequency_spectrum(frequency_range, magnitude_power, band1_indices, band2_indices, infile)

# Adjust layout for frequency spectrum subplots
plt.tight_layout()

# Establish figure for LF/HF ratios across activities
figure_count = p3m.establish_figure(figure_count)

# Plot LF/HF ratios
plt.bar(activities, lf_hf_ratios, color=['blue', 'green', 'orange', 'red'])
plt.title('LF/HF Ratios for Different Activities')
plt.xlabel('Activity')
plt.ylabel('LF/HF Ratio')
plt.grid(axis='y')
plt.tight_layout()

#%% Saving Figures

plt.figure(1)
plt.savefig('Raw_ECG_Signals_for_All_Activities.pdf', dpi=300)

plt.figure(2)
plt.savefig('Concatenated_ECG_Signal.pdf', dpi=300)

plt.figure(3)
plt.savefig('FIR_Filter_Time_and_Frequency_Response.pdf', dpi=300)

plt.figure(4)
plt.savefig('Raw_versus_Filtered_Data_Comparison.pdf', dpi=300)

plt.figure(5)
plt.savefig('Heartbeat_Detection_for_All_Activities.pdf', dpi=300)

plt.figure(6)
plt.savefig('HRV_SDNN_for_Different_Activities.pdf', dpi=300)

plt.figure(7)
plt.savefig('Frequency_Spectrum_with_LF_and_HF_Bands.pdf', dpi=300)

plt.figure(8)
plt.savefig('LF_HF_Ratio_Comparison_Across_Activities.pdf', dpi=300)








