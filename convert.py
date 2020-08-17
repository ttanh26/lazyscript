from utils.helper_function import get_record, work_in_local, sample_recognize
# import os
import librosa as li    
from pydub import AudioSegment
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf


# path = './lazyscript/sample_audio/ES2016a.Mix-Headset.wav'
path = './lazyscript/sample_audio/TED_talk.wav'

AudioSegment.from_wav(path)[50000:150000].export('./lazyscript/sample_audio/ES2016a_1min_2nd.wav', 'wav')

fs = 16000 # Sample rate
duration = 30 # Duration of recording

# get_record(fs, duration)

# data, sr  = li.load('./output.wav', sr = fs)
# sf.write('./lazyscript/sample_audio/output1_clean.wav', data, sr, subtype= 'PCM_16')

# work_in_local(['./lazyscript/sample_audio/output.wav'], ['Speaker1'], transcript_path= './lazyscript/transcript/')