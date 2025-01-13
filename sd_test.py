import sounddevice as sd
import soundfile as sf
import numpy as np 

sd.default.device = 12
fs = 16000
sd.default.samplerate = fs

duration = 5*fs #seconds

rec = sd.rec(duration,channels=2)
sd.wait()
print(rec.shape)
sf.write('./audio_out.wav',rec,fs)
