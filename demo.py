import streamlit as st
from utils.helper_function import *
from utils.speech_segmentation_predict import *

PRETRAINED_PATH = './lazyscript/pretrained_model'
FILE_PATH = './test.wav'
FOLDER_PATH = './lazyscript/save_audio'
TRANSCRIPT_PATH = './lazyscript/transcript'

plt.style.use('seaborn')

st.title('Lazyscript - Transcript for lazy')

st.header('Introduction')

st.markdown('''Generally, voice speed would be faster than writing speed. 
                It seems hard for audiences to write the full details of what they heard in a meeting. 
                Some people (include me), often make an audio record to hear again later. 
                By doing that, we can make sure that we would not miss any details in the meeting. 
                By applying Deep Learning models for making transcript automatically would save time and increase efficiency for us''')


st.header('Sample Audio')

base_path = './lazyscript/sample_audio/'

path = base_path + st.sidebar.selectbox('Select an audio file: ', os.listdir(base_path))

st.audio(path)

agree = st.sidebar.checkbox('Enable transcript', False)
if agree:
    st.header('Loading model')
    with st.spinner('Loading model. Wait for it...'):
        model = speech_segmentation_model(PRETRAINED_PATH)
    st.success('Done!')

    st.header('Preparing for transcript ...')
    with st.spinner('Processing audio...'):
        _, sample_rate = sf.read(path)
        seg_point = multi_segmentation(path, model= model, sample_rate= sample_rate)
        order_list = prepare_transcript(path, FOLDER_PATH, seg_point= seg_point)
        labels = cluster_seg(order_list, min_cluster= 2, max_cluster= 100, n_features= 125, n_mfcc= 12)
    st.success('Done!')

    st.header('Transcript')
    with st.spinner('Transcripting...'):
        destination_blob_name= path.split('/')[-1]
        work_in_cloud(order_list, labels, transcript_path= TRANSCRIPT_PATH ,sample_rate= sample_rate)
    st.success('Done')

    with open(os.path.join(TRANSCRIPT_PATH, 'transcript_125.txt'), 'r') as f:
        text = f.readlines()
    st.write(text)
else:
    st.header('Visualizing')

