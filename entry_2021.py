#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
from utils import pan_tompkin, comp_cosEn, save_dict


def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def ngrams_rr(data, length):
    grams = []
    for i in range(0, length-12, 12):
        grams.append(data[i: i+12])
    return grams

def ngrams_seq(data, length, t_unit):
    grams = []
    for i in range(0, length-t_unit, t_unit):
        grams.append(data[i: i+t_unit])
    return grams

def challenge_entry(sample_path):
    """

    """

    sig, _, fs = load_data(sample_path)
    sig = sig[:, 1]
    y_seq = np.zeros((len(sig), ), dtype=np.int)
    end_points = []

    _, _, r_peaks, _ = pan_tompkin(sig, fs=200)
    rr_seq = np.diff(r_peaks) / fs
    len_rr = len(rr_seq)

    t_unit = int(fs * 0.5)
    rr_seq_slice = ngrams_rr(rr_seq, len_rr)
    is_af = []
    for rr_period in rr_seq_slice:
        cos_en, _ = comp_cosEn(rr_period)
        if cos_en <= -1.4:
            is_af.append(0)
        else:
            is_af.append(1)
    is_af = np.array([[j] * 12 for j in is_af]).flatten()
    rr_seq_last = rr_seq[-12: ]
    cos_en, _ = comp_cosEn(rr_seq_last)
    if cos_en <= -1.4:
        is_af_last = 0
    else:
        is_af_last = 1
    
    len_rr_remain = len_rr - int(12*len(rr_seq_slice))
    is_af = np.concatenate((is_af, np.array([is_af_last] * len_rr_remain).flatten()), axis=0)
    
    if np.sum(is_af) == 0:
        y_class = 0
        y_seq = ngrams_seq(y_seq, len(y_seq), t_unit)
        y_seq = np.sum(y_seq, axis=-1).flatten()
        y_seq = y_seq > 0
    elif np.sum(is_af) == len(is_af):
        y_class = 1
        y_seq += 1
        y_seq = ngrams_seq(y_seq, len(y_seq), t_unit)
        y_seq = np.sum(y_seq, axis=-1).flatten()
        y_seq = y_seq > 0
    else:
        y_class = 2
        state_diff = np.diff(is_af)
        start_r = np.where(state_diff==1)[0] + 1
        end_r = np.where(state_diff==-1)[0] + 1
        if is_af[0] == 1:
            start_r = np.insert(start_r, 0, 0)
        if is_af[0] == 1:
            end_r = np.insert(end_r, len(end_r), len(is_af)-1)
        start_end = np.concatenate((start_r, end_r), axis=-1).tolist()
        end_points.append(start_end)
        for [start, end] in end_points:
            y_seq[r_peaks[start]: r_peaks[end]] += 1
        y_seq = ngrams_seq(y_seq, len(y_seq), t_unit)
        y_seq = np.sum(y_seq, axis=-1).flatten()
        y_seq = (y_seq > 0).astype(int)
    
    pred_dcit = {'predict_sequence': y_seq, 'predict_endpoints': end_points, 'predict_class': y_class}
    
    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    test_set = open(os.path.join(DATA_PATH, RECORDS), 'r').read().splitlines()
    for i, sample in enumerate(os.listdir(test_set)):
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

