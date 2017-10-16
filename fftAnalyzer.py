"""######################################################################

Miles Adamson
06/21/2017

Analyzes all wav files in directory set in save_path variable. 
Produces FFT plots, Time domain plots, Mode plots, csv's with data

######################################################################"""

import fftManager
import matplotlib
import matplotlib.pyplot as plt
from scipy import *
from numpy import *
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
import os
import time
start_time = time.time()

# Edit these properties for each run
save_folder = "C:\\mywavs"
block_name = "Wire supports alone"
conditions = "Plucking the wire support"

position = 5          # Starting position of strike in mm
strike_step = 5       # Step size between strikes in mm
tplusminus = 0.08     # Time on either side of hammer time we want to view
min_f = 500           # minimum frequency which we consider for peaks
delta = 50            # Data points on either side of local peak to consider
alpha = 0.0016        # Factor for steepness of peak
mode_threshold = 0.4  # Defines how commonly peaks need to appear to build a mode
mode_num = 6          # Number of modes to show on comparison plot

# True -> show some FFT plots with Python, not just png files
debugging = False

##############################################################################
save_path = save_folder + "\\Results\\"
modes_path = save_folder + "\\Modes\\"
plots_path =save_folder + "\\Signals\\"
folder = save_folder + "\\Wav Files\\"

mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(save_path))
name_list = os.listdir(folder)

peaks = []
peak_amps = []
largest_peaks = []
norm_peak_amps = []
peak_indexes = []
fft = []
wavs = []
FWHMs = []

plt.ioff()
        
# Get all wav file names. Only look at .wav files
for name in name_list:
    if str(name).endswith('.wav'):
        wavs.append(name)
wavs = sorted(wavs)
numfiles = len(wavs)
positions = [x * strike_step + position for x in range(0, numfiles)]

# Save info on each strike in a csv
csv_path = save_path + 'Summary.csv'
############################################################################## 
# Analyze all wav files and store that in a list of fftManager objects
for i, name in enumerate(wavs):
    
    path = folder + name
    # Create an fftManager Object from wav file
    fft.append(fftManager.fftManager(path, 0))
    
    # Process .wav
    fft[i].gently_massage(tplusminus, delta, alpha, min_f)
    
    peaks.append(fft[i].peak_freq)
    peak_amps.append(fft[i].peak_amplitudes)
    largest_peaks.append(fft[i].biggest_peaks_f)  
    norm_peak_amps.append(fft[i].peak_normalized_amp)
    peak_indexes.append(fft[i].peak_indexes)
    FWHMs.append(fft[i].FWHM)  
############################################################################## 
# Find the largest fft amplitude in the data set for plotting
maxis = []
for val in fft:
    maxis.append(max(val.fft_amp))
    
max_fft_amp = max(maxis)    

############################################################################## 
# Record everything in csv files

with open(csv_path, "wb") as csv_file:
    # Header for csv summary 
    writer = csv.writer(csv_file, delimiter=',')
    row = ["Cement Block", block_name]
    writer.writerow(row)
    row = ["Conditions", conditions]
    writer.writerow(row)
    row = ["Strike info below"]
    writer.writerow(row)

    for i, name in enumerate(wavs):
        
        mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(plots_path))
        # Plot and save the time domain signal
        fig = plt.figure(figsize=(20.0, 12.0))
        plt.plot(fft[i].time, fft[i].signal)
        
        #title = "Signal from Strike at " + str(position) + " mm"
        title = str(name) + " Signal"        
        
        plt.title(title)
        plt.ylabel('Signal (normalized to [+1/-1])')
        plt.xlabel('Time (s)')
        savefig(title + '.png', bbox_inches='tight')
        plt.close(fig)
          
        # Plot and save the f domain with red bars at each peak
        fig = plt.figure(figsize=(20.0, 12.0))
        axes = plt.gca()
        axes.set_ylim([0.0, max_fft_amp])
        axes.set_xlim([0.0, 17500.0])
        plt.plot(fft[i].fft_freq, fft[i].fft_amp)
        for p in peaks[i]:
            plt.axvline(p, color='r')
            
        #title = "FFT from Strike at " + str(position) + " mm"
        title = str(name) + " FFT"
        plt.title(title)
        plt.ylabel('FFT Amplitude')
        plt.xlabel('Frequency (Hz)')
        savefig(title + '.png', bbox_inches='tight')
        plt.close()

        mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(save_path))
        # Add entry for each strike
        writer = csv.writer(csv_file, delimiter=',')
        
        #row = ["\nStrike Index", str(i)]
        row = ["\nStrike Index", str(name)]
        
        writer.writerow(row)
        #myList = ','.join(map(str, str(positions[i])) 
        row = ["Strike Location" , positions[i]]
        writer.writerow(row)
        row = ["Number of Peaks", len(peaks[i])]
        writer.writerow(row)
        
        row = ["Peak Frequencies"]
        for j in peaks[i]:
            row.append(j)
        writer.writerow(row)
        
        row = ["Amplitude at Peaks"]
        for j in peak_amps[i]:
            row.append(j)
        writer.writerow(row)
        
        row = ["Normalized Amplitude at Peaks"]
        for j in norm_peak_amps[i]:
            row.append(j)
        writer.writerow(row)
        
        row = ["Peak FWHM's"]
        for j in FWHMs[i]:
            row.append(j)
        writer.writerow(row)
                

        # DEBUGGING PLOTS FOR SETTING PEAK ALPHA AND DELTA
        if debugging and i < 3:
            figure(i)
            plt.plot(fft[i].fft_freq, fft[i].fft_amp)
            for p in peaks[i]:
                plt.axvline(p, color='r')
            plt.show()
            
        # End of loop updates
        position += strike_step 

    csv_file.close()  
############################################################################## 
# Save ALL raw data into a CSV. First row is frequencies, 
# every row after is the raw fft amplitude data
    
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(save_path))   
csv_path = save_path + 'Raw Data.csv'
with open(csv_path, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(fft[0].fft_freq)
    for each_fft in fft:
        writer.writerow(each_fft.fft_amp) 
    csv_file.close()  
##############################################################################      
# Save all peak frequencies to a csv
        
csv_path = save_path + 'Peaks.csv'
with open(csv_path, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i, each_fft in enumerate(fft):
        info = ["Strike number:", i, "Strike location:", positions[i]]
        writer.writerow(info)
        writer.writerow(each_fft.peak_amplitudes)
        writer.writerow(each_fft.peak_normalized_amp)
        writer.writerow(each_fft.peak_freq)  
        writer.writerow(each_fft.peak_indexes)
        writer.writerow(each_fft.FWHM)
        writer.writerow("\n")
    csv_file.close()    
##############################################################################       
# Plot of location of where the largest peak was
        
fig = plt.figure(figsize=(30.0, 20.0))
plt.plot(positions, largest_peaks)
title = "Largest FFT Peak vs Position of Strike"
plt.title(title)
plt.xlabel('Strike Location (mm)')
plt.ylabel('Frequency of Largest Peak')
savefig(title + '.png', bbox_inches='tight')
plt.close(fig)



##############################################################################
# Modes

mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(modes_path))
# Sum up where all the peaks were throughout fft's
occurances = zeros(len(fft[0].fft_freq))
for p in peak_indexes:
    for index in p:
        occurances[index] += 1        

# Find the most commonly appearing peaks, defined by mode_threshold
common_modes = []
common_modes_f = []
for index, val in enumerate(occurances):
    if val >= max(occurances) * mode_threshold:
        common_modes.append(index)
        common_modes_f.append(fft[0].fft_freq[index])

# Stack all fft's into a 2d grid
fft_grid = zeros(len(fft[0].fft_freq))
for f in fft:
    fft_grid = np.vstack((fft_grid, f.fft_amp))

# Remove the zeros    
fft_grid = np.asarray(fft_grid[1:])

# Plot all of the most commonly appearing modes with their strength 
# at each location
count = 0
for mode in common_modes:
    if count < mode_num:
        y = fft_grid[:,mode]
        fig = plt.figure(figsize=(20.0, 12.0))
        plt.plot(positions, y)
        f = int(fft[0].fft_freq[mode])
        title = "Mode at " + str(f) + " Hz"
        plt.ylabel('FFT Amplitude')
        plt.xlabel('Strike Location (mm)')
        plt.title(title)
        savefig(title + '.png', bbox_inches='tight')
        plt.close(fig)
        count += 1


# Plot the occurances of each peak with a line at the threshold
fig = plt.figure(figsize=(20.0, 12.0))
plt.plot(fft[0].fft_freq, occurances)
z = zeros(len(occurances))
line = []
for val in occurances:
    line.append(max(occurances) * mode_threshold)
plt.plot(fft[0].fft_freq, line, color = 'r')
title = "Occurances of Peaks Across " + str(len(fft)) + " FFT's"
plt.ylabel('Occurances')
plt.xlabel('Frequency (Hz)')
plt.title(title)
savefig(title + '.png', bbox_inches='tight')
plt.close(fig)


# Plot all modes on one plot
c =  ['r', 'b', 'g', 'k', 'c', 'y', 'm', 'r', 'b', 'g', 'k', 'c', 'y', 'm']
fig = plt.figure(figsize=(20.0, 12.0))
count = 0

for n, mode in enumerate(common_modes):
    if count < mode_num:
        y = fft_grid[:,mode]
        f = int(fft[0].fft_freq[mode])
        l = str(f) + " Hz"
        plt.plot(positions, y, linestyle='None', marker='o', color = c[n], label=l)
        plt.legend(loc='upper left')
        title = "All Modes Found"
        plt.ylabel('FFT Amplitude')
        plt.xlabel('Strike Location (mm)')
        plt.title(title)
        savefig(title + '.png', bbox_inches='tight')
        count += 1
plt.close(fig)


##############################################################################
# 3d Plot all fft plots created side by side
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(save_path))
fig = plt.figure(figsize=(30.0, 20.0))
ax = fig.gca(projection='3d')
x = fft[0].fft_freq
index = 0
for i in fft:
    y = zeros(len(x)) + positions[index]
    z = i.fft_amp   
    ax.plot(x, y, z, label='curve')
    index += 1
plt.yticks(positions)  
title = "FFTs Across Measurements"
plt.title('FFTs Across Cylinder')
plt.ylabel('Strike Location (mm)')
plt.xlabel('Frequency (Hz)')
savefig(title + '.png')
plt.show()

##############################################################################


print "Done. Time in minutes:"
print((time.time() - start_time) / 60.0)

