import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import argmax
import soundfile as sf

# Function to calculate normalized cross-correlation between two signals
def correlation(Signal1, Signal2):
    # Compute the standard deviation (sigma) of the signals
    sigma1 = np.sqrt(np.correlate(Signal1, Signal1))
    sigma2 = np.sqrt(np.correlate(Signal2, Signal2))
    CrossCorr = np.correlate(Signal1, Signal2, mode="full") / (sigma1 * sigma2)

    # Compute the normalized cross-correlation
    return CrossCorr

# Function to estimate the angle of arrival using 
# GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
def GCC_phat(Mics, Signals, fs):

    # Check if the number of microphones matches the number of signals
    if not (len(Mics) == len(Signals)):
        print("The amount of mic and signals not at the same size")
        return -1
    
    # Speed of sound in air (m/s)
    c = 343

    # Find the time delay (in samples) between the two signals 
    delta_n = np.argmax(correlation(Signals[0], Signals[1])) - (Signals[1].size-1)

    # Convert the delay from samples to time (seconds)
    delta_t = delta_n /fs

     # Calculate the distance between the two microphones
    mic_dist = np.sqrt((Mics[0][0] -Mics[1][0])**2 +(Mics[0][1] -Mics[1][1])**2)

    # Check if the time delay is physically plausible
    if(np.abs(delta_t) * c < mic_dist):
        # Compute the angle/direction of arrival (DOA) in degrees
        theta = np.acos((delta_t*c)/mic_dist)
        return theta * 180/np.pi
    
     # If the delay is too large for the given microphone spacing, print an error
    print("The delay diffrance between the signals is to big for these micropones indexses")
    print("Check the micropones  indexses")
    return -1

if __name__ == '__main__':
    # wav,fs=sf.read('C:/Users/ido26/Documents/vscode projects\Localizetion project/first.wav')
    # res = GCC_phat(wav[:1024,:],fs)

    # Example microphone positions (in meters)
    mic1 = np.array([-1, 0.0]) #meters
    mic2 = np.array([1, 0.0]) #meters

    f_talk = 4000 #[Hz]
    fs = 16000 #16,000 Hz
    n = np.arange(0, 1024)
    #sig1 = np.sin(f_talk *2 * (np.pi) * n /fs)
    #sig2 = np.sin(f_talk * 2 * (np.pi) * (n+67) / fs)
    sig1 = np.zeros(1024)
    sig2 = np.zeros(1024)
    sig1[512] = 1
    sig2[512 - 67] = 1
    res = GCC_phat(np.array([mic1, mic2]), np.array([sig1, sig2]), fs )
    print (res)

    cor = correlation(sig1, sig2)
    n2 = range(cor.size)

    plt.plot(n2, cor)
    plt.show()


