#!/usr/bin/env python3

import numpy as np
import math
import json
import os
import sys

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import wfdb


class RefInfo():
    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.fs, self.len_sig, self.beat_loc, self.af_starts, self.af_ends, self.class_true = _load_ref()
        self.t_unit = int(self.fs * 0.5)
        self.y_true = _gen_y_seq() 
        if self.class_true == 1:
            self.onset_score_range, self.offset_score_range = _gen_endpoint_score_range()
        else:
            self.onset_score_range, self.offset_score_range = None, None

    def _load_ref(self):
        sig, fields = wfdb.rdsamp(self.sample_file)
        ann_ref = wfdb.rdann(self.sample_file, 'atr')

        fs = fields['fs']
        length = len(sig)
        sample_descrip = fields['comments']

        beat_loc = ann_ref.sample # r-peak locations
        ann_note = ann_ref.aux_note # rhythm change flag

        af_start_scripts = np.where((ann_note=='(AFIB') or (ann_note=='(AFL'))[0]
        af_end_scripts = np.where(ann_note=='(N')[0]

        if 'non atrial fibrillation' in sample_descrip:
            class_true = 0
        elif 'paroxysmal atrial fibrillation' in sample_descrip:
            class_true = 1

        elif 'persistent atrial fibrillation' in sample_descrip:
            class_true = 2
        else:
            print('Error: the recording is out of range!')

            return -1

        return fs, length, beat_loc, af_start_scripts, af_end_scripts, class_true

    def _ngrams_seq(self):
        grams = []
        for i in range(0, self.len_sig-self.t_unit, self.t_unit):
            grams.append(data[i: i+self.t_unit])

        return grams

    def _gen_y_seq(self):
        seq = np.zeros((self.len_sig, ), dtype=int)
        for [start, end] in zip(self.af_starts, self.af_ends):
            seq[start: end] += 1
        y_true = self._ngrams_seq(seq, self.len_sig, self.t_unit)
        y_true = np.sum(y_true, axis=-1).flatten()
        y_true = (y_true > 0).astype(int)

        return y_true
    
    def _gen_endpoint_score_range(self):
        af_beat_counts = self.af_ends - self.af_starts
        wg = lambda x: 1 if x > 20 else (math.log(20) / math.log(x))
        weights = list(map(wg, af_beat_counts))

        """
        [0, 1]: 4;
        [1, 2]: 2;
        [2, 3]: 1;
        [3, 5]: 0;
        [5: ]: -0.5
        """
        onset_range = np.ones((self.len_sig, ),dtype=np.float) * -0.5
        offset_range = np.ones((self.len_sig, ),dtype=np.float) * -0.5
        for i, af_start in enumerate(self.af_starts):
            onset_range[self.beat_loc[af_start-1]: self.beat_loc[af_start+2]] += 2.5
            onset_range[self.beat_loc[af_start-2]: self.beat_loc[af_start-1]] += 1.5
            onset_range[self.beat_loc[af_start+2]: self.beat_loc[af_start+3]] += 1.5
            onset_range[self.beat_loc[af_start-3]: self.beat_loc[af_start-2]] += 1
            onset_range[self.beat_loc[af_start+3]: self.beat_loc[af_start+4]] += 1
            onset_range[self.beat_loc[af_start-5]: self.beat_loc[af_start-3]] += 0.5
            onset_range[self.beat_loc[af_start+4]: self.beat_loc[af_start+6]] += 0.5
            onset_range[self.beat_loc[af_start-3]: self.beat_loc[af_start+4]] *= weights[i]
        for i, af_end in enumerate(self.af_ends):
            offset_range[self.beat_loc[af_end-1]: self.beat_loc[af_end+2]] += 2.5
            offset_range[self.beat_loc[af_end-2]: self.beat_loc[af_end-1]] += 1.5
            offset_range[self.beat_loc[af_end+2]: self.beat_loc[af_end+3]] += 1.5
            offset_range[self.beat_loc[af_end-3]: self.beat_loc[af_end-2]] += 1
            offset_range[self.beat_loc[af_end+3]: self.beat_loc[af_end+4]] += 1
            offset_range[self.beat_loc[af_end-5]: self.beat_loc[af_end-3]] += 0.5
            offset_range[self.beat_loc[af_end+4]: self.beat_loc[af_end+6]] += 0.5
            offset_range[self.beat_loc[af_end-3]: self.beat_loc[af_end+4]] *= weights[i]
        
        return onset_range, offset_range
    
def load_ans(ans_file):
    with open(ans_file, "r") as json_file:
        ans_dic = json.load(json_file)
    
    y_pred = ans_dic['predict_sequence']
    endpoints_pred = ans_dic['predict_endpoints']
    class_pred = ans_dic['predict_class']

    return y_pred, endpoints_pred, class_pred

def auprc_paro_af(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1, sample_weight=None)
    auprc = auc(recall, precision)

    return auprc

def paroaf_endpoint_score(endpoints_pred, onset_score_range, offset_score_range):
    score = 0
    for [start, end] in endpoints_pred:
        score += onset_score_range[int(start)]
        score += offset_score_range[int(end)]

    return score

def f_measurement_class(class_true, class_pred):
    A = np.zeros((3, 3), dtype=np.int)
    for c_true, c_pred in zip(class_true, class_pred):
        A[int(c_true), int(c_pred)] += 1
    
    f10 = 2 * A[0, 0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    f11 = 2 * A[1, 1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    f12 = 2 * A[2, 2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))

    f1 = (f10+f11+f12) / 3

    return f1

def score(data_path, ans_path):
    # AF burden estimation
    AUPRC_LOCAL = []
    TRUE_CLASS = []
    PRED_CLASS = []
    ENDPOINTSCORE = []

    test_set = open(os.path.join(data_path, 'RECORDS'), 'r').read().splitlines()
    for i, test_sample in enumerate(test_set):
        sample_path = os.path.join(data_path, test_sample)
        y_pred, endpoints_pred, class_pred = load_ans(os.path.join(ans_path, test_sample+'.json'))
        TrueRef = RefInfo(sample_path)

        TRUE_CLASS.append(TrueRef.class_true)
        PRED_CLASS.append(class_pred)
        
        if TrueRef.class_true == 1:
            auprc = auprc_paro_af(TrueRef.y_true, y_pred)
            AUPRC_LOCAL.append(auprc)
            endpoint_score = paroaf_endpoint_score(endpoints_pred, TrueRef.onset_score_range, TrueRef.offset_score_range)
            ENDPOINTSCORE.append(endpoint_score)
    
    global_f1 = f_measurement_class(TRUE_CLASS, PRED_CLASS)
    af_event_score = 0.5 * global_f1 + 0.5 * np.mean(AUPRC_LOCAL)
    af_endpoint_score = np.sum(ENDPOINTSCORE)

    return af_event_score, af_endpoint_score


if __name__ == '__main__':
    af_event_score, af_endpoint_score = score(sys.argv[1], sys.argv[2])
    print('AF Event Screen Performance: %0.4f' %af_event_score)
    print('AF Endpoints Detection Performance: %0.4f' %af_endpoint_score)

    with open('score.txt', 'w') as score_file:
        print('AF Event Screen Performance: %0.4f' %af_event_score, file=score_file)
        print('AF Endpoints Detection Performance: %0.4f' %af_endpoint_score, file=score_file)

        score_file.close()
