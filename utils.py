import sys
import numpy as np

import pandas as pd
from sklearn import preprocessing
from scipy import signal as sig


def pan_tomken(data, fs):
    N = 24

    rE = fs//3

    x = data.astype("float")

    x = (x - np.mean(x)) / np.std(x)

    x1 = sig.lfilter([1,0,0,0,0,0,-2,0,0,0,0,0,1],[1,-2,1],x)
    x2 = sig.lfilter([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[1,1],x1)
    x3 = np.zeros(x.shape)
    for i in range(2,len(x2)-2):
        x3[i] = (-1*x2[i-2] -2*x2[i-1] + 2*x2[i+1] + x2[i+2])/(8*T)        
    x4 = x3*x3
    x5 = np.zeros(x.shape)
    for i in range(N,len(x4)-N):
        for j in range(N):
            x5[i]+= x4[i-j]
    x5 = x5/N

    peaki = x5[0]
    spki = 0
    npki = 0
    peak = [0]
    threshold1 = spki
    pk = []
    for i in range(1,len(x5)):
        if x5[i]>peaki:
            peaki = x5[i]

        npki = ((npki*(i-1))+x5[i])/i
        spki = ((spki*(i-1))+x5[i])/i
        spki = 0.875*spki + 0.125*peaki
        npki = 0.875*npki + 0.125*peaki

        threshold1 = npki + 0.25*(spki-npki)
        threshold2 = 0.5 * threshold1

        if(x5[i]>=threshold2):

            if(peak[-1]+N<i):
                peak.append(i)
                pk.append(x5[i])

    p = np.zeros(len(x5))
    rPeak = []

    for i in peak:
        if(i==0 or i<2*rE):
            continue
        p[i]=1

        ind = np.argmax(x2[i-rE:i+rE])
        maxIndexR = (ind+i-rE)
        rPeak.append(maxIndexR)

    rPeak = np.unique(rPeak)

    return rPeak

def sampen(rr_seq, max_temp_len, r):
    """
    rr_seq: segment of the RR intervals series
    max_temp_len: maximum template length
    r: initial value of the tolerance matching
    """ 
    length = len(rr_seq)
    lastrun = np.zeros((1,length))
    run = np.zeros((1,length))
    A = np.zeros((max_temp_len,1))
    B = np.zeros((max_temp_len,1))
    p = np.zeros((max_temp_len,1))
    e = np.zeros((max_temp_len,1))

    for i in range(length - 1):
        nj = length - i - 1
        for jj in range(nj):
            j = jj + i + 2
            if np.abs(rr_seq[j-1] - rr_seq[i]) < r:
                run[0, jj] = lastrun[0, jj] + 1
                am1 = float(max_temp_len)
                br1 = float(run[0,jj])
                M1 = min(am1,br1)
				
                for m in range(int(M1)):
                    A[m] = A[m] + 1
                    if j < length:
                        B[m] = B[m]+1
            else:
                run[0, jj] = 0

        for j in range(nj):
            lastrun[0, j] = run[0,j]

    N = length * (length - 1) / 2
    p[0] = A[0] / N
    e[0] = -1 * np.log(p[0] + sys.float_info.min)
    for m in range(max_temp_len-1):
        p[m+1]=A[m+1]/B[m]
        e[m+1]=-1*np.log(p[m+1])

    return e, A, B

def comp_cosEn(rr_segment):
    r = 0.03 # initial value of the tolerance matching
    max_temp_len = 2 # maximum template length
    min_num_count = 5 # minimum numerator count
    dr = 0.001 # tolerance matching increment
    match_num = np.ones((max_temp_len,1)) # number of matches for m=1,2,...,M
    match_num = -1000 * match_num
    while match_num[max_temp_len-1,0] < min_num_count:
        e, match_num, B = sampen(rr_segment, max_temp_len, r)
        r = r + dr
    if match_num[max_temp_len-1, 0] != -1000:
        mRR = np.mean(rr_segment)
        cosEn = e[max_temp_len-1, 0] + np.log(2 * (r-dr)) - np.log(mRR)
    else:
        cosEn = -1000
    sentropy = e[max_temp_len-1, 0]

    return cosEn, sentropy

def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)