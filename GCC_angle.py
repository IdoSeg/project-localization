import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import argmax
import soundfile as sf
import itertools as iter

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

def sampel_estimetion(cords, speaker, fs):
    c = 34300 # cm/s
    NofChannel = len(cords)
    PN = int(NofChannel * (NofChannel - 1)/2) # number of total possible combination
    kp = np.array(list(iter.combinations(range(NofChannel), 2))) 
    # kp is a martix that 
    # any row combains to number that represent mic pair

    dist = []
    dist_calc = 0
    for point in cords:
        dist_calc = np.linalg.norm(speaker - point)
        dist.append(dist_calc)


    sam_evl = np.zeros(PN)
    for i in range(0, PN):
        k1 = kp[i, 0]
        k2 = kp[i, 1]
        sam_evl[i] = (dist[k2] - dist[k1]) * fs / c
        print("The sampel estimation for the(", k1+1, ", ", k2+1, ") pair is: ", sam_evl[i])
    return sam_evl

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
    #print (res)

    cor = correlation(sig1, sig2)
    n2 = range(cor.size)

    # plt.plot(n2, cor)
    # plt.show()

    # ---------------------------------------------------------------------------------
    # exemple 3 -> white, brown, talk
    
    d = 4.57 # cm
    d_2 = d * np.sqrt(2)

    mic1 = np.array([d_2/2, 0.0, 0.0]) # (x, y, z)
    mic2 = np.array([0.0, d_2/2, 0.0])
    mic3 = np.array([-d_2/2, 0.0, 0.0])
    mic4 = np.array([0.0, -d_2/2, 0.0])
    mics = np.array([mic1, mic2, mic3, mic4]) 
    speaker = np.array([d_2/2 + 15, 0.0, 0.0])

    print("For the first exempel(white, brown, talk) the estimetions are: ")
    A = sampel_estimetion(mics, speaker, 16000)

    # exemple 4 -> pink, green

    mic1 = np.array([d, d, 0.0]) 
    mic2 = np.array([0.0, d, 0.0])
    mic3 = np.array([d, 0.0, 0.0])
    mic4 = np.array([0.0, 0.0, 0.0]) 
    mics = np.array([mic1, mic2, mic3, mic4]) 
    speaker = np.array([d + 15, d, 0.0])

    print()
    print("For the second exempel(pink, green) the estimetions are: ")
    B = sampel_estimetion(mics, speaker, 16000)
    



