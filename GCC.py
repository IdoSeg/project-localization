import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import argmax
import soundfile as sf

def correlation(Signal1, Signal2):
    sigma1 = np.sqrt(np.correlate(Signal1, Signal1))
    sigma2 = np.sqrt(np.correlate(Signal2, Signal2))
    CrossCorr = np.correlate(Signal1, Signal2, mode="full") / (sigma1 * sigma2)
    return CrossCorr

def GCC_phat(Mics, Signals, fs):
    if not (len(Mics) == len(Signals)):
        print("The amount of mic and signals not at the same size")
        return -1
    c = 343 #m/s
    delta_n = np.argmax(correlation(Signals[0], Signals[1])) - (Signals[1].size-1)
    delta_t = delta_n /fs
    mic_dist = np.sqrt((Mics[0][0] -Mics[1][0])**2 +(Mics[0][1] -Mics[1][1])**2)
    if(np.abs(delta_t) * c < mic_dist):
        theta = np.acos((delta_t*c)/mic_dist)
        return theta * 180/np.pi
    print("The delay diffrance between the signals is to big for these micropones indexses")
    print("Check the micropones  indexses")
    return -1

if __name__ == '__main__':
    # wav,fs=sf.read('C:/Users/ido26/Documents/vscode projects\Localizetion project/first.wav')
    # res = GCC_phat(wav[:1024,:],fs)
    mic1 = np.array([-1, 0.0]) #meters
    mic2 = np.array([1, 0.0]) #meters
    f_talk = 4000 #[Hz]
    fs = 2**14 #16000 Hz
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


