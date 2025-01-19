import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import argmax
import itertools as iter
import soundfile as sf
from scipy.signal import resample_poly
import sounddevice as sd

def GCC_features_one_frame(x, fs, tua_grid ,upSampeling = 1):
    '''
    Computes Generalized Cross-Correlation (GCC) features for a single frame of multichannel audio data.

    Args:
        x (numpy.ndarray): A 2D array of shape (N_sample, NOfChann) representing the multichannel audio data,
                           where N_sample is the number of time-domain samples, and NOfChann is the number of channels.
        fs (integer): The sampling frequency of the audio data.
        tua_grid (numpy.ndarray): A 1D array representing the delay grid (time lags) for cross-correlation computation.

    Returns:
        numpy.ndarray: A 2D array of shape (N_tua, PN), where N_tua is the length of tua_grid and PN is the number of
                       microphone pairs. Each column corresponds to the GCC features for a specific microphone pair.
                       so the returned value is a 2D array which every colmn represent a corrletion of a two signals pair

    Notes:
        - The function computes the GCC features by calculating the phase transform (PHAT) of the cross-power
          spectrum between all microphone pairs.
    '''

    # varribles
    NOfChann = len(x[0]) #number of channels //2
    N_sample = len(x) #number of samples //1024
    N_freq = int(np.ceil(N_sample/2))+1 #number of rellevent frequencys //513
    
    #tua_grid = np.arange(-25, 26)/fs #correlletion shifts axis 
    N_tua = len(tua_grid) #//51 as a defult
    channalNum = np.arange(0, NOfChann) # array between 1 to number of sinals
    PN = int(NOfChann * (NOfChann - 1)/2) #number of total possible combination
    
    kp = np.array(list(iter.combinations(channalNum, 2))) # kp is a martix that any row combains to number that represent mic pair
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
        P = P / (np.abs(P) + np.finfo(np.float64).eps) # added mechin epsilon to prevent divding by 0 in an extreme case

        spec = np.zeros((N_freq, N_tua ))
        for ind in range(0, N_tua):
            EXP = np.exp(-2j * np.pi * (tua_grid[ind] / upSampeling) * f.T)
            spec[:,ind] = np.real(P * EXP)
        feast[kk,:] = np.sum(spec, axis=0)  
    return feast.T 

def GCC_features_full_signals(x, fs, nfft = 1024, nhop = 512, N_half_tua = 25, upSampeling = 1):
    '''
     Computes Generalized Cross-Correlation (GCC) features for full multichannel audio signals.

    Args:
        x (numpy.ndarray): A 2D array of shape (N_sample, NOfChann), where N_sample is the number of time-domain 
        samples, and NOfChann is the number of audio channels.
        fs (int): The sampling frequency of the audio data.
        nfft (int, optional): The FFT size for windowed signal processing. Default is 1024.
        nhop (int, optional): The hop size for windowed signal processing. Default is 512.
        N_half_tua (int, optional): Half the number of time-delay grid points for cross-correlation. Default is 25.
        upSampeling (int, optional): Upsampling factor for the input signals. Default is 1 (no upsampling).

    Returns:
        numpy.ndarray: A 3D array of shape (N_frames, N_tua, PN), where:
            - N_frames: Number of frames processed in the signal.
            - N_tua: Number of time-delay grid points (2 * N_half_tua + 1).
            - PN: Number of microphone pairs (NOfChann choose 2).

    Notes:
        - The function up-samples the input signal for geting a highier angel resulotion
        - Zero-padding is applied to ensure the total signal length is a multiple of the hop size for simplicity.
        - Features are computed for each frame of the signal using `GCC_features_one_frame`, which processes
          individual frames of audio data.
    '''

    # up sampeling
    # x, fs = interpolate(x, fs, upSampeling, 1)

    # varibles
    tua_grid = np.arange(-N_half_tua, N_half_tua + 1)/fs #correlletion shifts axis
    N_sample  = len(x)
    NOfChann = len(x[0]) #number of channels //2
    N_tua = len(tua_grid)
    PN = int(NOfChann * (NOfChann - 1)/2) #number of total possible combination

    # padding with zeros until we get a munifactor of nhop
    zer =  np.zeros((nhop - (N_sample % nhop), NOfChann))
    x = np.append(x, zer, axis=0)
    N_frames = int(N_sample / nhop - 1)
    N_sample  = len(x)

    # orgnized the data into featuers
    featurs = np.zeros(shape = (N_frames, N_tua, PN))
    for frame in range(0, N_frames):
        start = frame * (nfft - nhop)
        end = start + nfft
        send = x[start:end]
        #send, fs_int = interpolate(send, fs, upSampeling, 1)
        featurs[frame, :, :] = GCC_features_one_frame(send, fs, tua_grid, upSampeling)
    
    return featurs

def max_element_tua(features, Ntua, upSampeling = 1):
    '''
     Computes the estimeted delay using max arg of the correletion time vector for each frame, pair in features.

    Args:
        features (numpy.ndarray): A 3D array of shape (N_frames, N_tua, PN)

    Returns:
        numpy.ndarray: A 2D array of integers values represents the estimeted delay for each (N_frames, PN)
    '''
    max_indices = np.argmax(features, axis=1)
    return (max_indices - Ntua)/upSampeling

def interpolate(x, fs, upFactor, downfactor):
    '''
    resample x by factor of (upFactor/ downfactor) = resampling factor

    Args:
        x (numpy.ndarray): A 2D array of shape (N_sample, NOfChann)

    Returns:
        numpy.ndarray: A 2D array of shape (N_sample * resampling factor , NOfChann)
        
    Notes:
        - The function up-samples the input signal withe the perpose of geting a highier angel resulotion
        - The function has the flexibility to also use down sample factor even though it is not in use 
    '''
    up_fs = fs * upFactor
    # Resample
    upsampled_signal = resample_poly(x, upFactor, downfactor, axis=0)
    return [upsampled_signal, up_fs]

if __name__ == '__main__':
    # Exemples

    # # exemple 1 -> recording the signals using the 2 comuter mics   
    # audio_out,fs1 = sf.read("C:/Users/ido26/Documents/vscode projects/Localizetion project/audio_out.wav")
    # n = np.arange(-25,26)
    # res = GCC_features_full_signals(audio_out, fs1, 1024, 512, 25, 2)
    # delays = max_element_tua(res, 25)

    # # ploting remdom frame coreletion
    # plt.plot(n, res[44,:])
    # plt.xlabel('Time - n * fs')
    # plt.ylabel('correletion value')
    # plt.title('R(n*fs)')
    # plt.show()
    
    # # ploting the max correletion value - frame graph
    # plt.plot(delays)
    # plt.xlabel('Time - n * frame size')
    # plt.ylabel('max arg correletion value')
    # plt.title('Max arg, represent the estimated delay')
    # plt.show()

    #-----------------------------------------------------------------------------------------------------------------------

    # exemple 2 -> recording the signals using the output device, 6 channels
    audio_6_outputs, fs2 = sf.read("C:/Users/ido26/Documents/vscode projects/Localizetion project/audio_out_4_mics.wav")
    
    # soundcheck
    # try_s , fs_up  = interpolate(audio_6_outputs, fs2, 8, 1)
    # sd.play(try_s[:,1] ,fs_up )
    # sd.wait()
    
    n = np.arange(-25,26)
    res = GCC_features_full_signals(audio_6_outputs[:, 1:5], fs2, 1024, 512, 25)
    delays = max_element_tua(res, 25)

    # ploting remdom frame coreletion
    # plt.plot(n, res[44,:, 0]) #rendom frame(44), the first pair correletion
    # plt.xlabel('Time - n * fs')
    # plt.ylabel('correletion value')
    # plt.title('Rendom frame corrletion values')
    # plt.show()


    # ploting the max correletion value - frame graph
    # plt.plot(delays[:,0]) # the delays for the second pair
    # plt.xlabel('Time in frames ')
    # plt.ylabel('max arg correletion value')
    # plt.title('pair (0,1) represent the estimated delay')
    # plt.show()

    #-----------------------------------------------------------------------------------------------------------------------
    
    # exemple 3, 4 - checking interpoletions possibels

    exmple3, fs = sf.read("white_noise_out.wav")
    # "white_noise_out.wav", "brown_noise_out.wav" , "talk_noise_out.wav"
    # "pink_noise_out.wav", "green_noise_out.wav" 

    res = GCC_features_full_signals(exmple3[:, 1:5], fs, 1024, 512, 25, 4)
    delays = max_element_tua(res, 25, 4)

    plt.subplot(2, 3, 1)
    plt.plot(delays[:, 0])
    plt.title("(1, 2)")

    plt.subplot(2, 3, 2)
    plt.plot(delays[:, 1])
    plt.title("(1, 3)")

    plt.subplot(2, 3, 3)
    plt.plot(delays[:, 3])
    plt.title("(1, 4)")

    plt.subplot(2, 3, 4)
    plt.plot(delays[:, 3])
    plt.title("(2, 3)")

    plt.subplot(2, 3, 5)
    plt.plot(delays[:, 3])
    plt.title("(2, 4)")

    plt.subplot(2, 3, 6)
    plt.plot(delays[:, 3])
    plt.title("(3, 4)")

    plt.suptitle("The max correlletion results")
    plt.show()

    plt.plot(n, res[44,:, 1]) # rendom frame(44), the first pair correletion
    plt.show()



