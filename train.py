# from helper_function_local import *
# import os
# import librosa as li    
# from pydub import AudioSegment
# from pydub.utils import make_chunks
from DeepSpeech.speech_segmentation_predict import *
from DeepSpeech.test import *


def work_in_cloud(order_list, labels, transcript_path, sample_rate = 16000):
    config = {'language_code': 'en-US'}
    # config.update({'enable_automatic_punctuation': True})
    # file_path ='./sample_LDC.wav'
    bucket_name = 'tata_jff'
    transcript_path = os.path.join(transcript_path, 'transcript_125.txt')
    text_list = []

    for idx, file_path in enumerate(order_list):
        destination_blob_name = file_path.split('/')[-1]
        audio = {'uri': 'gs://tata_jff/' + destination_blob_name}
        upload_object(bucket_name= bucket_name, file_path= file_path, destination_blob_name= destination_blob_name)
        text = speech_to_text_long(f'gs://{bucket_name}/{destination_blob_name}', labels[idx], transcript_path, sr = sample_rate)
        text_list.append(text)

    texting = '\n'.join(text_list)
    with open(transcript_path, 'w') as f:
        f.write(texting)

# if __name__ == "__main__":
    # audio = sph_to_wav(TED_sample)

    # AudioSegment.from_wav('./ES2011d.Mix-Headset.wav')[:300000].export('ES2011d_5mins_1st.wav', 'wav')

    # order_list, labels = run_option('./ES2011d_5mins_1st.wav', True)
    # work_in_cloud(order_list, labels)
