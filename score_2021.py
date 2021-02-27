#!/usr/bin/env python3

import numpy as np
import math
import os
import sys
import glob

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


def load_ref(sample_file):
    sig, fields = wfdb.rdsamp(sample_file)
    ann_ref = wfdb.rdann(sample_file. 'atr')

    y_true = np.zeros((len(sig), ), dtype=int)
    weights = np.ones((len(sig), ), dtype=int)

    ann_loc = ann_ref.sample
    ann_type = ann_ref.symbol
    ann_note = ann_ref.aux_note

    if len(rhythm_change_indices) == 0:

        return y_true, weights
    else:
        ref_start_scripts = np.where(ann_note=='(AFIB')[0]
        ref_end_scripts = np.where(ann_note=='(N')[0]

        for [ref_start_script, ref_end_script] in zip(ref_start_scripts, ref_end_scripts):            
            positive_count = len(ann_type[ref_start_script: ref_end_script]) - 1
            y_true[ann_loc[ref_start_script]: ann_loc[ref_end_script]] += 1

            if positive_count > 20:
                continue
            else:
                weights[ann_loc[ref_start_script]: ann_loc[ref_end_script]] *= (math.log(20) / math.log(positive_count))

        return y_true, weights


def score(ref_path, ans_path):
    # AF burden estimation
