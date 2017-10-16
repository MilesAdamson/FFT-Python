"""######################################################################

Miles Adamson
5/31/2017

fftManager takes a signal and its sample rate in its contructor. It then
calculates a corresponding time signal and dt. Contains methods to create
fft's and plot things. Largely based off of jim's code: jimfft2.py

Methods include:

__init__(self, signal, sample_rate)
Constructor takes either a signal and its sample rate OR a path to a wav file 
and then any placeholder for the 2nd input

pad(pad_size, sides)
zero pads the signal and updates time array to match

pad_default()
Takes the signal and doubles its length with padded zeros on the end.
Then, continues to zero pad until the length is a power of 2

normalize()
normalizes the given signal to a [-1, 1] range

fft()
creates power and amplitude fft's and corresponding frequency array

plot_signal(title)
plots time domain signal with given title

plot_amplitude_fft(title)
plots amplitude fft with given title

plot_power_fft(title)
plots power fft with given title

crop_time(self, t_min, t_max)
Crops signal and time array to be in between two times (inclusive).
Then updates things like hammer_time and signal_len

crop_fft(self, f_low, f_high)
Crops fft_amp and fft_power to be in between two frequencies (inclusive).

peaks2(delta, alpha, min_f)
Finds the peaks in FFT based on the number of data points on either side of 
it to consider (delta), the steepness of peak (alpha) and a minimum frequency

gently_massage(t, delta, alpha, min_f)
Processes the time domain signal based on the time to either side of
impulse (t), and the inputs to be given to peak2


######################################################################"""
from scipy import *
from numpy import *
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
import pylab as plt
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

class fftManager(object):
    
    # All variables which this class contains, initialized to zero or empty
    signal = array([], 'float')
    time = array([], 'float')
    fft = array([], 'float')
    fft_freq = array([], 'float')
    fft_amp = array([], 'float')
    fft_power = array([], 'float')
    peak_freq = array([], 'float')
    peak_indexes = array([], 'float')
    peak_amplitudes = array([], 'float')
    peak_normalized_amp = array([], 'float')   
    FWHM = array([], 'float')   
    
    biggest_peaks_f = 0
    peak_num = 0
    dt = 0
    sample_rate = 0
    fft_len = 0
    fft_len_amp = 0
    signal_len = 0
    padded_fft_len = 0
    hammer_time = 0
    hammer_time_index = 0
    
######################################################################    
# Initializer takes in a signal and its sample rate. Calculate some
# basics off of that info like dt and its corresponding time array
    
    def __init__(self, signal, sample_rate):
  # def __init__(self, C:\dir\file_name, 0):

        # given a file name and directory, assumes two channel input
        if type(signal) is str:
            # Example string: "c:\\WAV File\\Block 4.wav"
            wav_file_info = wavfile.read(signal)
            self.sample_rate = wav_file_info[0]  
            self.dt = 1.0 / self.sample_rate   
            ch1 = array(wav_file_info[1][:,0], "float")
            ch2 = array(wav_file_info[1][:,1], "float")
            self.signal = (ch1 + ch2) / 2.0      
            self.signal_len = len(self.signal)
            self.__make_time()
            
        # given a signal and its sample rate
        else:
            self.signal = signal
            self.sample_rate = sample_rate
            
            # Set basics from constructor
            self.signal_len = len(self.signal)
            self.dt = 1.0 / self.sample_rate  
            self.__make_time()
        
######################################################################  
# Pad takes in a number of zeros to pad with a string which shows it
# which sides of the signal to add them: at the start, end or both.
# Updates the time array to match the new signal.
        
    def pad(self, pad_size, sides):
        # Create list of zeros of lenth pad_size
        pad = array([], 'float')
    
        pad = np.zeros(pad_size) 
        # Do nothing with a negative pad size                     
        if pad_size < 0:
            pass
        
        # Add pad to the start of the signal
        elif sides == "start" or sides == "Start":
            self.signal = append(pad, self.signal)
            self.signal_len = len(self.signal)
            self.__make_time()
    
        # Add pad to the end of the signal
        elif sides == "end" or sides == "End":
            self.signal = append(self.signal, pad)
            self.signal_len = len(self.signal)
            self.__make_time()
    
        # Add pad to both sides of the signal
        elif sides == "both" or sides == "Both":
            temp = append(pad, self.signal)
            self.signal = append(temp, pad)
            self.signal_len = len(self.signal)
            self.__make_time()
            
        # No pad
        elif sides == "none" or sides == "None":   
            pass
    
        # Incorrect pad mode string
        else:
           pass
       
######################################################################  
# Pad the end of a signal with zeros to make the signal legnth a power 
# of 2 in length. Significantly increases fft speed for large signals. 
# Around half of the signal should be zeros for fft (padded at the end)

    def pad_default(self):
        
        power_of_2 = 2
        
        # find the first power of 2 > signal length
        for i in range(1, 100):
            if power_of_2 <= self.signal_len: 
                power_of_2 = power_of_2 * 2
            elif power_of_2 >= self.signal_len:
                break
            
        # Zero pad the signal to be power_of_2 in length
        self.pad(power_of_2 - self.signal_len, "end")
       
###################################################################### 
# __make_time creates an array of time stamps matching the current signal
           
    def __make_time(self):
        
        empty = array([], 'float')
        for i in range(len(self.signal)):
            empty = append(empty, i * self.dt)
            
        self.time = empty
            
###################################################################### 
# fft takes the current signal and creates an FFT for it. Also creates
# an array of fft amplitudes and powers
    
    def fft(self):
        
        self.fft = fft(self.signal)
        self.fft_len = len(self.fft)
        self.fft_amp = 2.0 / self.fft_len * abs(self.fft[0 : self.fft_len/2])
        self.fft_power = self.fft_amp*self.fft_amp
        self.fft_freq = linspace(0.0, 1.0 / (2.0 * self.dt), self.fft_len/2)
        self.fft_len_amp = len(self.fft_amp)
    
###################################################################### 
# Normalize the data in signal to [-1, 1] range   
    
    def normalize(self):
        
        norm = max(abs(self.signal))                  
        self.signal = self.signal/norm
        pass
        
        
######################################################################
# Plots the time domain signal vs time
            
    def plot_signal(self, title):
        
        plt.title(title)
        plt.xlabel("t (s)")
        plt.ylabel("Amplitude")
        plt.plot(self.time, self.signal, 'k-')
        plt.show()     
        
######################################################################
# Plots the fft amplitudes vs freq
            
    def plot_amplitude_fft(self, title):
        
        plt.title(title)
        plt.xlabel("f (Hz)")
        plt.ylabel("Amplitude")
        plt.plot(self.fft_freq, self.fft_amp, 'k-')
        plt.show()    
        
######################################################################
# Plots the fft power vs freq
            
    def plot_power_fft(self, title):
        
        plt.title(title)
        plt.xlabel("f (Hz)")
        plt.ylabel("Power")
        plt.plot(self.fft_freq, self.fft_power, 'k-')
        plt.show()     
        
######################################################################
# Crops all fft variables to a given frequency range
        
    def crop_fft(self, f_low, f_high):
        
        filt_a = array([], 'float')
        filt_p = array([], 'float')
        filt_f = array([], 'float')
        
        for i in range(self.fft_len_amp):
            if self.fft_freq[i] >= f_low and self.fft_freq[i] <= f_high:
                filt_f = append(filt_f, self.fft_freq[i])
                filt_a = append(filt_a, self.fft_amp[i])
                filt_p = append(filt_p, self.fft_power[i])
                    
        self.fft_amp = filt_a
        self.fft_power = filt_p
        self.fft_freq = filt_f
        self.signal_len = len(self.signal)
    
######################################################################
# Crops all time variables to a given time range.
        
    def crop_time(self, t_min, t_max):

        filt_v = array([], 'float')
        filt_t = array([], 'float')
        for i in range(self.signal_len):
            if self.time[i] >= t_min and self.time[i] <= t_max:
                filt_t = append(filt_t, self.time[i])
                filt_v = append(filt_v, self.signal[i])
                    
        self.signal = filt_v
        self.time = filt_t
        self.signal_len = len(self.signal)
 
######################################################################
# Finds the "on" point for an impulse response type signal. Adapted from 
# from jimfft2.py, and added functionality to get the index from the
# time array, not just the time value itself
 
    def find_hammer_time(self):
        
        t = self.time
        v = self.signal
        diff = array([], 'float')

        for i in range(len(v)-1):
            diff = append(diff, abs(v[i+1]-v[i]))
            
        diff = append(diff, 0.0)
        mval = max(diff)/2.0        # Threshold value.
        j_on = 0
        test = diff[0]
        
        while test < mval:
            j_on = j_on + 1
            test = diff[j_on] 
            
        self.hammer_time = t[j_on]
        self.hammer_time_index = int(self.hammer_time / self.dt)
        
######################################################################
# Get all peaks above a threshold percentage of max value in the amplitude
# fft. Set the peak frequencies and their respective indexes in self.fft_freq
# Only finds peaks above min_f input. Call fft() before this method
        
# Suggested values:
# delta = 4       Data points on either side of local peak to consider
# alpha = 0.005   Factor for steepness of peak relative to max in fft
# beta = 16       Factor to increase delta by for averaging window size
# zeta = 2        Factor for averaging window
# min_f 2000      First peak seems to always be at 3.5k
 
    def peaks(self, delta, alpha, beta, zeta, min_f):
        v = self.fft_amp
        maxv = max(v)
        indexes = arange(len(v))     
        # step through indexes of V, [0 to 4096] or whatever
        for i in indexes:
            # do not go outside of signal array to negative indexes
            if i - (delta + beta) >= 0:
                # do not go past the end of signal array
                if i + (delta + beta) <= len(indexes):
                    w = v[i - delta : i + delta + 1]
                    failed = 0
                    # can't be a peak if not center of window
                    if max(w) != v[i]:
                        failed += 3
                    else:
                        w_indexes = range(len(w))
                        first_half = w_indexes[:int(len(w)/2) + 1]
                        second_half = w_indexes[int(len(w)/2):]
                        # Ramps up for delta data points before peak?
                        mini = -Inf

                        for j in first_half:
                            if w[j] >= mini:
                                mini = w[j]
                            else:
                                failed += 1
                                
                        # Ramps down for delta data points after peak?
                        maxi = Inf
                        for j in second_half:
                            if w[j] <= maxi:
                                maxi = w[j]
                            else:
                                failed += 1
                                   
                        # Is this data point too close to the average value nearby?
                        l = v[i - delta - beta: i + delta + beta + 1]     
                        ave = sum(l) / float(len(l))
                        if v[i] / ave < zeta:
                            failed += 3
                    
                    # Data point i in v is a peak if we didn't fail tests
                    if failed < 2:
                        peak_amp = v[i] - ((w[0] + w[-1]) / 2)
                        relative_peak_amp = peak_amp / maxv
                        if relative_peak_amp > alpha:
                            if self.fft_freq[i] > min_f:
                                self.peak_freq = append(self.peak_freq, self.fft_freq[i])
                                self.peak_amplitudes = append(self.peak_amplitudes, v[i])
                                self.peak_indexes = append(self.peak_indexes, i)                     
        # Record the largest peak
        biggest_peak = self.peak_amplitudes.tolist().index(max(self.peak_amplitudes))
        self.biggest_peaks_f = self.peak_freq[biggest_peak]
        
        for amp in self.peak_amplitudes:
            k = amp / max(self.peak_amplitudes)
            self.peak_normalized_amp = append(self.peak_normalized_amp, k) 
            
        # Calculate FWHM
        for i in self.peak_indexes:
            amp = self.fft_amp[i]
            left_half = amp
            right_half = amp
            j = i
            while left_half > amp / 2:
                left_half = self.fft_amp[j]
                j -= 1
                # If it wasn't found, do not go to negative indexes
                if j == 0:
                    break
            k = i    
            while right_half > amp / 2:
                right_half = self.fft_amp[k]
                k += 1
                # If it wasn't found, do not go outside array
                if k == len(self.fft_amp):
                    break
            self.FWHM = append(self.FWHM, self.fft_freq[k] - self.fft_freq[j])     

######################################################################
# Massage signal to be normalized, shorter, zero padded to be 2 ^ n in 
# length. Then fft() it. t is the amount of time on either side of hammer
# time we wish to analyze. Threshold is the % of max fft amplitude value
# we call a peak. min_f is the minimum freq which we will count a peak.
     
    def gently_massage(self, t, delta, alpha, min_f):
        self.find_hammer_time()
        self.crop_time(self.hammer_time - t, self.hammer_time + t)
        self.normalize()
        self.pad_default()
        self.fft()
        self.peaks2(delta, alpha, min_f)
        
                    
#########################################################################
                    
    def peaks2(self, delta, alpha, min_f):
        v = self.fft_amp
        f = self.fft_freq
        """
        if above f minimum..
        if local maximum within delta controlled window..
        if sufficiently larger than average data to the side..
        It's a peak
        """        
        for i, val in enumerate(v):
            if f[i] > min_f:   
                if i - delta >=0 and i + delta + 1 <= len(v):
                    w = v[i - delta : i + delta + 1]
                    wl = v[i - delta : i]
                    wr = v[i : i + delta + 1]
                    if max(w) == val:  
                        al = sum(wl)/float(len(wl))
                        ar = sum(wr)/float(len(wr))
                        if (val - al) > alpha or (val - ar) > alpha:
                            self.peak_freq = append(self.peak_freq, f[i])
                            self.peak_amplitudes = append(self.peak_amplitudes, v[i])
                            self.peak_indexes = append(self.peak_indexes, i)               
         
        # If we found no peaks return 0
        if len(self.peak_amplitudes) == 0:
            self.peak_freq = append(self.peak_freq, 0.0)
            self.peak_amplitudes = append(self.peak_amplitudes, 0.0)
            self.peak_indexes = append(self.peak_indexes, 0.0) 
            return 
            
        # Record the largest peak
        biggest_peak = self.peak_amplitudes.tolist().index(max(self.peak_amplitudes))
        self.biggest_peaks_f = self.peak_freq[biggest_peak]
        
        for amp in self.peak_amplitudes:
            k = amp / max(self.peak_amplitudes)
            self.peak_normalized_amp = append(self.peak_normalized_amp, k) 
            
        # Calculate FWHM
        for i in self.peak_indexes:
            amp = self.fft_amp[i]
            left_half = amp
            right_half = amp
            j = i
            while left_half > amp / 2:
                left_half = self.fft_amp[j]
                j -= 1
                # If it wasn't found, do not go to negative indexes
                if j == 0:
                    break
            k = i    
            while right_half > amp / 2:
                right_half = self.fft_amp[k]
                k += 1
                # If it wasn't found, do not go outside array
                if k == len(self.fft_amp):
                    break
            self.FWHM = append(self.FWHM, self.fft_freq[k] - self.fft_freq[j])
                    
#########################################################################
                
                
                
                
        