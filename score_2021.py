#!/usr/bin/env python3

import numpy as np
import math
import os
import sys
import glob

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import wfdb

from entry_2021 import challenge_entry
from utils import load_dict


def ngrams_seq(data, length, t_unit):
    grams = []
    for i in range(0, length-t_unit, t_unit):
        grams.append(data[i: i+t_unit])
    return grams


def gen_y_seq(seq, t_unit, af_starts, af_ends):
    for [start, end] in zip(af_starts, af_ends):
        seq[start: end] += 1
    y_true = ngrams_seq(seq, len(seq), t_unit)
    y_true = np.sum(y_true, axis=-1).flatten()
    y_true = (y_true > 0).astype(int)

    return y_true


def load_ref(sample_file):
    sig, fields = wfdb.rdsamp(sample_file)
    ann_ref = wfdb.rdann(sample_file, 'atr')

    fs = fields['fs']
    sample_descrip = fields['comments']
    t_unit = int(fs * 0.5)

    y_seq = np.zeros((len(sig), ), dtype=int)
    y_true = np.zeros((len(sig) // t_unit, ), dtype=int)

    # ann_loc = ann_ref.sample # r-peak locations
    ann_note = ann_ref.aux_note # rhythm change flag

    af_start_scripts = np.where((ann_note=='(AFIB') or (ann_note=='(AFL'))[0]
    af_end_scripts = np.where(ann_note=='(N')[0]
    weights = np.ones((len(af_start_scripts), ), dtype=int)
    af_beat_counts = af_end_scripts - af_start_scripts
    end_points_true = np.concatenate((af_start_scripts, af_end_scripts), axis=-1).tolist()

    
    if 'non atrial fibrillation' in sample_descrip:
        class_true = 0
    elif 'paroxysmal atrial fibrillation' in sample_descrip:
        class_true = 1
        y_true = gen_y_seq(y_seq, af_start_scripts, af_end_scripts, t_unit)

    elif 'persistent atrial fibrillation' in sample_descrip:
        class_true = 2
        y_true += 1
    else:
        print('Error: the recording is out of range!')

        return -1

    wg = lambda x: 1 if x > 20 else (20 / x)
    weights = list(map(wg, af_beat_counts))

    return y_true, end_points_true, class_true, weights


def auprc_paro_af(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1, sample_weight=None)
    auprc = auc(recall, precision)

    return auprc

def f_measurement_class(class_true, class_pred):
    A = np.zeros((3, 3), dtype=np.int)
    for c_true, c_pred in zip(class_true, class_pred):
        A[int(c_true)][int(c_pred)] += 1
    
    f10 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    f11 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    f12 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))

    f1 = (f10+f11+f12) / 3

    return f1

def score_endpoints(endpoints_true, endpoints_pred, weights):
    

def score(data_path, ans_path):
    # AF burden estimation
    test_set = open(os.path.join(data_path, RECORDS), 'r').read().splitlines()
    AUPRC_LOCAL = []
    TRUE_CLASS = []
    PRED_CLASS = []
    TRUE_ENDPOINTS = []
    PRED_ENDPOINTS = []
    for i, test_sample in enumerate(test_set):
        sample_path = os.path.join(data_path, test_sample)
        y_true, endpoints_true, class_true, weights = load_ref(sample_path)
        result = load_dict(os.path.join(ans_path, test_sample))
        y_pred = result['predict_sequence']
        endpoints_pred = result['predict_endpoints']
        class_pred = result['predict_class']

        if class_true == 1:
            auprc = auprc_paro_af(y_true, y_pred)
            AUPRC_LOCAL.append(auprc)

        TRUE_CLASS.append(class_true)
        PRED_CLASS.append(class_pred)
        TRUE_ENDPOINTS.append(endpoints_true)
        PRED_ENDPOINTS.append(endpoints_pred)







