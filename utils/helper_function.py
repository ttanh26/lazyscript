import speech_recognition as sr
import soundfile as sf
import pandas as pd 
import random 
from pydub import AudioSegment
import os
from sklearn.cluster import KMeans
import numpy as np 
from spectralcluster import SpectralClusterer
import librosa as li
from google.cloud import storage
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
import sounddevice as sd
import io


def get_path(src):
    folder_path = os.scandir(src)
    path_list = []
    
    for x in folder_path:
        if x.is_file():
            path_list.extend([x.path])
        elif (x.is_dir()) and (x.name != '.ipynb_checkpoints'):
            path_list.extend(get_path(x))

    return path_list

def get_record(fs, duration):
    print('Start recording...')
    my_recording = sd.rec(int(fs * duration), samplerate= fs, channels= 1)
    sd.wait()
    print('Done!')
    sf.write('./lazyscript/sample_audio/output.wav', my_recording, fs, subtype= 'PCM_16')

def prepare_transcript(path, folder_path, seg_point):
    """
    Split the audio file into smaller one based on speech segmentation model.
    Args:
        path: original file
        folder_path: folder to save outputs
        seg_point: timestamp to spilt audio file
    Return:
        order_list: list of path of splited audio files 
    """
    audio = AudioSegment.from_file(path)
    order_list = []
    start = 0

    for idx, t in enumerate(seg_point):
        if idx == len(seg_point):
            break
        
        end = t * 1000
        # print(f'split at {start} - {end} ms')
        audio_chunk = audio[start: end]
        audio_chunk.export(f'{folder_path}/audio_chunk{idx}.wav', format= 'wav')
        order_list.append(f'{folder_path}/audio_chunk{idx}.wav')
        start = end
    end = len(audio) * 1000
    # print(f'split at {start} - {len(audio) * 1000} ms')
    audio_chunk = audio[start: end]
    audio_chunk.export(f'{folder_path}/audio_chunk{idx + 1}.wav', format= 'wav')
    order_list.append(f'{folder_path}/audio_chunk{idx + 1}.wav')

    return order_list


# Big problem here is by using BiLSTM model in speech_segmentation_predict.py to segment speaker-changing point,
def cluster_seg(order_list, min_cluster, max_cluster, n_features, n_mfcc):
    """
    Clustering small audio files based on their feature vectors (MFCCs) for speaker verification
    Args:
        order_list: list of splited audio files
        min_cluster: minimum number of cluster
        max_cluster: maximum number of cluster
        n_features: No. feature be chose for clustering
        n_mfcc: No. of MFCCs features
    Return:
        labels: Speaker verification for each audio files.
    """
    # Create an empty matrix with fixed-size (n_sample, n_feature)
    n_feat = np.zeros((len(order_list), n_features * n_mfcc))

    for idx, audio in enumerate(order_list):
        data, sr = li.load(audio, 16000)        
        # Calculating MFCCs features
        mfccs = li.feature.mfcc(data, sr, n_mfcc = n_mfcc)
        mfccs = np.array(mfccs)
        # Resize MFCCs and flatten 
        mfccs = np.resize(mfccs, (n_mfcc, n_features))
        mfccs = mfccs.flatten()

        n_feat[idx, :] = mfccs
    print(n_feat.shape)
    # Apply K-means to cluster audio files by their features vectors (MFCCs)    
    clusterer = SpectralClusterer(min_clusters= min_cluster, max_clusters= max_cluster)
    labels = clusterer.predict(n_feat)
    labels = [str(x) for x in labels]
    print(labels)
    # Convert from cluster labels to speaker identification
    speaker_tag = {}
    n = 1
    speaker_tag[labels[0]] = f'Speaker {n}: '
    for s in labels:
        if s not in speaker_tag:
            n += 1
            speaker_tag[s] = f'Speaker {n}: '
    labels = [speaker_tag[i] for i in labels]

    return labels


# def get_transcript(path_list, labels, transcript_path= './transcript_sample.txt'):
#     full_transcript = {}
#     err = 0
#     r = sr.Recognizer()
#     count_duration = 0
    
#     for idx, path in enumerate(path_list):
#         with sr.AudioFile(path) as source:
#             audio_data = r.record(source)
#             try:
#                 text = r.recognize_google(audio_data)
#                 full_transcript[f'{str(idx)} - {path}'] = labels[idx] + text
#             except:
#                 full_transcript[f'{str(idx)} - {path}'] = labels[idx] + '</br>'
#                 err += 1
#                 data, sample_rate = sf.read(path)
#                 duration = len(data)/sample_rate
#                 count_duration += duration
#     print(len(full_transcript), len(path_list), err, count_duration)
#     text_list = full_transcript.values()
#     text = '\n'.join(text_list)
#     print(text)
#     with open(transcript_path, 'w') as f:
#         f.write(text)
    
#     return text


def upload_object(bucket_name, file_path, destination_blob_name):
    """
    Upload a file to the bucket of google storage
    
    Args:
        bucket_name: cloud storage bucket name
        file_path: file_path in local
        destination_blob_name: save name on cloud storage
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(file_path)
    print(f'File {file_path} uploaded to {destination_blob_name}')


def speech_to_text_long(storage_uri, labels, transcript_path, sample_rate = 16000):
    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition

    Args:
        storage_uri: URI for audio files in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
        labels: list of labels for each audio file
        transcript_path: path to save transcript text file
        sr: sample_rate. Default = 16000 Hz = 16 kHz
    Return:
        text: transcript string
    """

    client = speech.SpeechClient()

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = sample_rate

    # The language of the supplied audio
    language_code = "en-US"

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "sample_rate_hertz": sample_rate_hertz,
        "language_code": language_code,
        "encoding": encoding,
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()
    text = labels
    
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        text += alternative.transcript 
        # print(u"Transcript: {}".format(alternative.transcript))

    return text


def print_sentences(response):
    for result in response.results:
        best_alter = result.alternatives[0]
        transcript = best_alter.transcript
        confidence = best_alter.confidence

        print('-' * 80)
        print(f'Transcript: {transcript}')
        print(f'Confidence: {confidence:.0%}')


def work_in_cloud(order_list, labels, transcript_path, sample_rate = 16000):
    """
    Main function to make transcript. Have 2 main parts: 
        Upload segmented audio files to cloud storage
        Get transcript through Google API from these files
    Args:
        order_list: list of path
        labels: label list of audio files
        transcript_path: place to save transcript text file
        sample_rate: Frequency of original audio. Default = 16000 Hz
    """
    # Set config parameter
    config = {'language_code': 'en-US'}
    bucket_name = 'tata_jff'
    transcript_path = os.path.join(transcript_path, 'transcript_125.txt')
    text_list = []

    for idx, file_path in enumerate(order_list):
        destination_blob_name = file_path.split('/')[-1]
        audio = {'uri': 'gs://tata_jff/' + destination_blob_name}

        # For audio has duration longer than 1 mins, we have to upload to cloud storage first
        upload_object(bucket_name= bucket_name, file_path= file_path, destination_blob_name= destination_blob_name)
        
        # Using Google Speech-to-text API to convert to text for each audio file we upload to cloud storage
        text = speech_to_text_long(f'gs://{bucket_name}/{destination_blob_name}', labels[idx], transcript_path, sample_rate= sample_rate)
        text_list.append(text)

    # Concatanating into a string and save the transcript into a text file
    texting = '\n'.join(text_list)
    with open(transcript_path, 'w') as f:
        f.write(texting)


def sample_recognize(local_file_path, labels):
    """
    Transcribe a short audio file using synchronous speech recognition

    Args:
      local_file_path Path to local audio file, e.g. /path/audio.wav
    """

    client = speech.SpeechClient()

    # local_file_path = 'resources/brooklyn_bridge.raw'

    # The language of the supplied audio
    language_code = "en-US"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
    }
    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = client.recognize(config, audio)
    text = labels
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        text += alternative.transcript + '\n'

    return text

def work_in_local(order_list, labels, transcript_path, sample_rate = 16000):
    transcript_path = os.path.join(transcript_path, 'transcript_125.txt')
    text_list = []

    for idx, file_path in enumerate(order_list):
        # For audio has duration longer than 1 mins, we have to upload to cloud storage first
        # Using Google Speech-to-text API to convert to text for each audio file we upload to cloud storage
        text = sample_recognize(file_path, labels[idx])
        text_list.append(text)

    # Concatanating into a string and save the transcript into a text file
    texting = '\n'.join(text_list)
    with open(transcript_path, 'w') as f:
        f.write(texting)

