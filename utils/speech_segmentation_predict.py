import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, TimeDistributed, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li
import os

def speech_segmentation_model(pre_trained_path, load_weights= True):
    """
    Load pretrained model
    """
    model_name = 'speech_seg'
    json_model_file = os.path.join(pre_trained_path, model_name + '.json')
    h5_model_file = os.path.join(pre_trained_path, model_name + '.h5')

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences= True)))
    model.add(Bidirectional(LSTM(128, return_sequences= True)))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(32)))
    model.add(TimeDistributed(Dense(1, activation= 'sigmoid')))

    model.build(input_shape= (None, 200, 35))
    # model.summary()

    model.load_weights(h5_model_file)
    return model

def multi_segmentation(file, model, sample_rate= 16000, frame_size= 1024, frame_shift= 256):
    data, sr = li.load(file, sr= sample_rate)
    mfccs = li.feature.mfcc(data, sr, n_mfcc = 12, hop_length= frame_shift, n_fft= frame_size)
    mfcc_delta = li.feature.delta(mfccs)
    mfcc_delta2 = li.feature.delta(mfccs, order= 2)

    mfcc = mfccs[1:, ]
    norm_mfcc= (mfcc - np.mean(mfcc, axis= 1, keepdims= True)) / np.std(mfcc, axis= 1, keepdims= True)
    norm_mfcc_delta = (mfcc_delta - np.mean(mfcc_delta, axis= 1, keepdims= True)) / np.std(mfcc_delta, axis= 1, keepdims= True)
    norm_mfcc_delta2 = (mfcc_delta - np.mean(mfcc_delta2, axis= 1, keepdims= True)) / np.std(mfcc_delta2, axis= 1, keepdims= True)

    ac_feature = np.vstack((norm_mfcc, norm_mfcc_delta, norm_mfcc_delta2))
    # print(ac_feature.shape)

    sub_seq_len = int(2.4 * sr/ frame_shift)
    sub_seq_step = int(0.6 * sr/frame_shift)

    def extract_feature():
        feature_len = ac_feature.shape[1]
        sub_train_x = []

        for i in range(0, feature_len - sub_seq_len, sub_seq_step):
            sub_seq_x = np.transpose(ac_feature[:, i: i + sub_seq_len])
            sub_train_x.append(sub_seq_x[np.newaxis, :, :])
        
        return np.vstack(sub_train_x), feature_len

    predict_x, feature_len = extract_feature()
    # print(predict_x.shape)

    predict_y = model.predict(predict_x)
    # print(predict_y.shape)

    score_acc = np.zeros((feature_len, 1))
    score_cnt = np.ones((feature_len, 1))

    for i in range(predict_y.shape[0]):
        for j in range(predict_y.shape[1]):
            idx = i * sub_seq_step + j
            score_acc[idx] += predict_y[i, j, 0]
            score_cnt[idx] += 1
    
    score_norm = score_acc/ score_cnt

    w_start = 0
    w_end = 150
    w_grow = 200
    delta = 25

    store_cp = []
    idx = 0
    while w_end < feature_len:
        score_seg = score_norm[w_start: w_end]
        max_v = np.max(score_seg)
        max_idx = np.argmax(score_seg)
        idx += 1
        if max_v > 0.3:
            temp = w_start + max_idx
            store_cp.append(temp)
            w_start += max_idx + 50
            w_end = w_start + w_grow
        else:
            w_end += w_grow

    seg_point = np.array(store_cp) * frame_shift

    # plt.figure('speech segmentation plot')
    # plt.plot(np.arange(0, len(data)) / float(sr), data, 'b-')

    # for i in range(len(seg_point)):
    #     plt.vlines(seg_point[i] / float(sr), -1, 1, colors= 'c', linestyles= 'dashed')
    #     plt.vlines(seg_point[i] / float(sr), -1, 1, colors= 'r', linestyles= 'dashed')

    # plt.xlabel('Time/s')
    # plt.ylabel('Speech Amp')
    # plt.grid(True)
    # plt.show()

    return np.asarray(seg_point) / float(sr)


