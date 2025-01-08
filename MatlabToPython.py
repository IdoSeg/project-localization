import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import argmax
import itertools as iter
import soundfile as sf

def GCC_features_one_frame(x, fs, tua_grid):
    # varribles
    NOfChann = len(x[0]) #number of channels //2
    N_sample = len(x) #number of samples //1024
    N_freq = int(np.ceil(N_sample/2))+1 #number of rellevent frequencys //513
    
    #tua_grid = np.arange(-25, 26)/fs #correlletion shifts axis 
    N_tua = len(tua_grid) #//51 as a defult
    channalNum = np.arange(0, NOfChann) #array between 1 to number of sinals
    PN = int(NOfChann * (NOfChann - 1)/2) #number of total possible combination
    
    kp = np.array(list(iter.combinations(channalNum, 2))) #kp is a martix that any row combains to number that represent mic pair
    feast = np.zeros((PN, N_tua)) #the final feateurs we send
    
    for kk in range(0, PN):
        kk1 = kp[kk, 0]
        kk2 = kp[kk, 1]
        
        kk1_spec = np.fft.fft(x[:, kk1])
        kk2_spec = np.fft.fft(x[:, kk2])
        
        freqs = np.fft.fftfreq(N_sample, 1/fs)
        f = freqs[0:N_freq]
        
        kk1_spec = kk1_spec[0:N_freq]
        kk2_spec = kk2_spec[0:N_freq]       

        P = kk1_spec * np.conj(kk2_spec)
        P = P / np.abs(P + np.finfo(np.float64).eps) #added mechin epsilon

        spec = np.zeros((N_freq, N_tua ))
        for ind in range(0, N_tua):
            EXP = np.exp(-2j * np.pi * tua_grid[ind] * f.T)
            spec[:,ind] = np.real(P * EXP)
        feast[kk,:] = np.sum(spec, axis=0)
    
    return feast.T 

def GCC_features_full_signals(x, fs, nfft, nhop, N_tua):
    tua_grid = np.arange(-N_tua, N_tua + 1)/fs #correlletion shifts axis
    N_sample  = len(x)
    NOfChann = len(x[0]) #number of channels //2
    PN = int(NOfChann * (NOfChann - 1)/2) #number of total possible combination

    # padding with zeros until we get a munifactor of nhop
    zer =  np.zeros((nhop - (N_sample % nhop), NOfChann))
    x = np.append(x, zer, axis=0)
    N_frames = int(N_sample / nhop - 1)
    N_sample  = len(x)
    
    featurs = GCC_features_one_frame(x[0:nfft], fs, tua_grid)
    for i in range(1, N_frames):
        start = i * nhop
        end = start + nfft
        featurs = np.append(featurs, GCC_features_one_frame(x[start:end], fs, tua_grid), axis=1)
        # rect window
    
    # featurs3D = featurs.reshape(2*N_tua+1, PN, N_frames) 
    # featurs3D = featurs3D.reshape(N_frames, 2*N_tua+1, PN)
    return featurs.T

def max_element_tua(features, Ntua):
    return [(np.argmax(row) - Ntua) for row in features]

if __name__ == '__main__':
    audio_out,fs1 = sf.read("C:/Users/ido26/Documents/vscode projects/Localizetion project/audio_out.wav")

    n = np.arange(-25,26)
    res = GCC_features_full_signals(audio_out, fs1, 1024, 512, 25)
    delays = max_element_tua(res, 25)

    plt.plot(n, res[44,:]) #may be with abs, and also in the max element
    plt.show()
    
    plt.plot(delays)
    plt.show()

