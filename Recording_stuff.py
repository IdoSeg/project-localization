import sounddevice as sd
import soundfile as sf
import numpy as np 

print("Available devices:")
print(sd.query_devices())

# Set default device by index (validate the index for your system)
device_index = 1  # Replace with the correct device index, -> indexes that works computer: 1,6, microphone array:aolso 1
sd.default.device = device_index

# Set sample rate and duration
fs = 16000  # Sampling rate in Hz //works for 44100, 2**16
duration = 5  # Recording duration in seconds

# Set samplerate and channels for recording
sd.default.samplerate = fs
sd.default.channels = 2  # Set the number of input channels (e.g., 1 for mono, 2 for stereo) //1,2,4 works

# Record audio
print("Recording...")
try:
    rec = sd.rec(int(duration * fs))  # Record for 'duration' seconds
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    rec = (rec/np.max(abs(rec),0))*0.9 
    # Save the recording as a WAV file
    #sf.write('./audio_out_4_mics.wav', rec, fs)
    sf.write('./audio_out.wav', rec, fs)
    print("Audio saved as 'audio_out.wav'.")
except Exception as e:
    print(f"An error occurred: {e}")

