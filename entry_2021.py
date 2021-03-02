#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
from utils import pan_tomken, comp_cosEn


def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def challenge_entry(data_path, ans_path):
    test_set = open(os.path.join(data_path, RECORDS), 'r').read().splitlines()
    for i, test_sample in enumerate(test_set):
        sample_path = os.path.join(data_path, test_sample)
        sig, _, fs = load_data(sample_path)
        r_peaks = pan_tomken(sig, fs=200)
        rr_seq = np.diff(r_peaks) / fs
        

        
    return y_score


if __name__ == '__main__':

