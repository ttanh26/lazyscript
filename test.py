from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
from google.cloud import storage
from pydub import AudioSegment

def speech_to_text(config, audio):
    client = speech.SpeechClient()
    response = client.long_running_recognize(config, audio)
    print_sentences(response)

def speech_to_text_long(storage_uri, labels, transcript_path, sr = 16000):
    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition

    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    client = speech.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = sr

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
    text = ''
    # print(response.results)
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        text += labels + alternative.transcript
        # text_list.append(labels + alternative.transcript)
        print(u"Transcript: {}".format(alternative.transcript))

    return text


def print_sentences(response):
    for result in response.results:
        best_alter = result.alternatives[0]
        transcript = best_alter.transcript
        confidence = best_alter.confidence

        print('-' * 80)
        print(f'Transcript: {transcript}')
        print(f'Confidence: {confidence:.0%}')

def upload_object(bucket_name, file_path, destination_blob_name):
    """Upload a file to the bucket of google storage
    # bucket_name = "storage-bucket-name"
    # file_path = "file-path-in-local"
    # destination_blob_name = "storage-object-name"
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(file_path)
    print(f'File {file_path} uploaded to {destination_blob_name}')

# file_path ='./test_set.wav'
# destination_blob_name = file_path.split('/')[-1]
# config = {'language_code': 'en-US'}
# config.update({'enable_automatic_punctuation': True})
# audio = {'uri': 'gs://tata_jff/' + destination_blob_name}
# bucket_name = 'tata_jff'


# if __name__ == "__main__":
    # sound = AudioSegment.from_wav(file_path)
    # sound = sound.set_channels(1)
    # sound.export('./test_set.wav', 'wav')
    # upload_object(bucket_name= bucket_name, file_path= file_path, destination_blob_name= destination_blob_name)
    # # speech_to_text(config, audio)
    # speech_to_text_long(f'gs://{bucket_name}/{destination_blob_name}','Speaker 1: ', './transcript_sample_1.txt')