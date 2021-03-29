import json
import numpy as np
import sys

import matplotlib.pyplot as plt
import pandas as pd
import peakutils
from sklearn import preprocessing
from scipy import signal

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn
"""

def p_t_qrs(ecg_original, fs=1000, gr=1):
    delay = 0
    skip = 0
    m_selected_RR = 0
    mean_RR = 0
    ser_back = 0 
    
    if (fs == 200):
        # Low pass and High pass
        # Low pass
        wn = 12 * 2 / fs
        N = 3                                                              
        a, b = signal.butter(N, wn, 'low')
        ecg_l = signal.filtfilt(a, b, ecg_original)
        ecg_l = ecg_l / max(abs(ecg_l))
        ecg_l = np.around(ecg_l, decimals=4)

        # High pass
        wn = 5 * 2 / fs
        N = 3                                          
        a, b = signal.butter(N, wn, 'high')                                        
        ecg_h = signal.filtfilt(a, b, ecg_original) 
        ecg_h = ecg_h / max(abs(ecg_h))

    else:
        # Bandpass
        f1 = 5                                                  
        f2 = 15
        wn = []
        wn.append(f1 * 2 / fs)
        wn.append(f2 * 2 / fs)
        N = 3                                                     
        a, b = signal.butter(N, wn, 'bandpass')                                                  
        ecg_h = signal.filtfilt(a, b, ecg_original)
        ecg_h = ecg_h / max(abs(ecg_h))

    # Derivative
    int_c = (5 - 1) / (fs * 1 / 40)
    x = np.arange(1,6)
    xp = np.dot(np.array([1, 2, 0, -2, -1]), (1 / 8) * fs)
    fp = np.arange(1,5+int_c,int_c)
    b = np.interp(fp, x, xp)
    ecg_d = signal.filtfilt(b, 1, ecg_h)
    ecg_d = ecg_d / max(ecg_d)

    # Squaring and Moving average
    ecg_s = np.power(ecg_d, 2)
    ecg_m = np.convolve(ecg_s ,np.ones(int(np.around(0.150*fs)))/np.around(0.150*fs))
    delay = delay + np.around(0.150*fs) / 2
    # Fiducial Marks
    locs = peakutils.indexes(ecg_m, thres=0, min_dist=np.around(0.2 * fs))
    pks = ecg_m[locs[:]]

    # Init other parameters
    LLp = len(pks)
    qrs_c = np.zeros(LLp)
    qrs_i = np.zeros(LLp)
    qrs_i_raw = np.zeros(LLp)
    qrs_amp_raw= np.zeros(LLp)
    nois_c = np.zeros(LLp)
    nois_i = np.zeros(LLp)
    SIGL_buf = np.zeros(LLp)
    NOISL_buf = np.zeros(LLp)
    SIGL_buf1 = np.zeros(LLp)
    NOISL_buf1 = np.zeros(LLp)
    THRS_buf1 = np.zeros(LLp)
    THRS_buf = np.zeros(LLp)

    # Init training phase
    THR_SIG = max(ecg_m[0:2*fs])*1/3
    THR_NOISE = np.mean(ecg_m[0:2*fs])*1/2
    SIG_LEV= THR_SIG
    NOISE_LEV = THR_NOISE


    # Init bandpath filter threshold
    THR_SIG1 = max(ecg_h[0:2*fs])*1/3
    THR_NOISE1 = np.mean(ecg_h[0:2*fs])*1/2
    SIG_LEV1 = THR_SIG1                      
    NOISE_LEV1 = THR_NOISE1

    # Thresholding and desicion rule
    Beat_C = -1
    Beat_C1 = -1
    Noise_Count = 0

    for i in range(LLp):
        if ((locs[i] - np.around(0.150*fs)) >= 1 and (locs[i] <= len(ecg_h))):
            _start = locs[i] - np.around(0.15*fs).astype(int)
            _ = ecg_h[_start:locs[i]]
            y_i = max(_)
            x_i = np.argmax(_)
        else:
            if i == 0:
                y_i = max(ecg_h[0:locs[i]])
                x_i = np.argmax(ecg_h[0:locs[i]])
                ser_back = 1
            elif (locs[i] >= len(ecg_h)):
                _ = ecg_h[locs[i] - np.around(0.150*fs).astype(int):]
                y_i = max(_)
                x_i = np.argmax(_)

        # Update the heart_rate    
        if (Beat_C >= 9):
            diffRR = np.diff(qrs_i[Beat_C-8:Beat_C])
            mean_RR = np.mean(diffRR)
            comp = qrs_i[Beat_C] - qrs_i[Beat_C-1]
            if ((comp <= 0.92*mean_RR) or (comp >= 1.16*mean_RR)):
                THR_SIG = 0.5*(THR_SIG)
                THR_SIG1 = 0.5*(THR_SIG1)               
            else:
                m_selected_RR = mean_RR

        # Calculate the mean last 8 R waves to ensure that QRS is not
        if m_selected_RR:
            test_m = m_selected_RR
        elif (mean_RR and m_selected_RR == 0):
            test_m = mean_RR
        else:
            test_m = 0

        if test_m:
            if ((locs[i] - qrs_i[Beat_C]) >= np.around(1.66*test_m)):
                _start = int(qrs_i[Beat_C] + np.around(0.20*fs))
                _end = int(locs[i] - np.around(0.20*fs))
                pks_temp = max(ecg_m[_start:_end+1])
                locs_temp = np.argmax(ecg_m[_start:_end+1])
                locs_temp = qrs_i[Beat_C] + np.around(0.20*fs) + locs_temp - 1

                if (pks_temp > THR_NOISE):
                    Beat_C += 1
                    qrs_c[Beat_C] = pks_temp
                    qrs_i[Beat_C] = locs_temp

                    if (locs_temp <= len(ecg_h)):
                        _start = int(locs_temp - np.around(0.150*fs))
                        _end = int(locs_temp + 1)
                        y_i_t = max(ecg_h[_start:_end])
                        x_i_t = np.argmax(ecg_h[_start:_end])
                    else:
                        _ = locs_temp - np.around(0.150*fs)
                        y_i_t = max(ecg_h[_:])
                        x_i_t = np.argmax(ecg_h[_:])

                    if (y_i_t > THR_NOISE1):
                        Beat_C1 += 1
                        qrs_i_raw[Beat_C1] = locs_temp - np.around(0.150*fs) + (x_i_t - 1)
                        qrs_amp_raw[Beat_C1] = y_i_t
                        SIG_LEV1 = 0.25*y_i_t + 0.75*SIG_LEV1

                    not_nois = 1
                    SIG_LEV = 0.25*pks_temp + 0.75*SIG_LEV       
            else:
                not_nois = 0

        # Find noise and QRS peaks
        if (pks[i] >= THR_SIG): 
            if (Beat_C >= 3):
                if ((locs[i] - qrs_i[Beat_C]) <= np.around(0.3600*fs)):
                    _start = locs[i] - np.around(0.075*fs).astype('int')
                    Slope1 = np.mean(np.diff(ecg_m[_start:locs[i]]))
                    _start = int(qrs_i[Beat_C] - np.around(0.075*fs))
                    _end = int(qrs_i[Beat_C])
                    Slope2 = np.mean(np.diff(ecg_m[_start:_end]))
                    if abs(Slope1) <= abs(0.5*(Slope2)):
                        nois_c[Noise_Count] = pks[i]
                        nois_i[Noise_Count] = locs[i]
                        Noise_Count += 1
                        skip = 1
                        NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1
                        NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV
                    else:
                        skip = 0

            if (skip == 0):
                Beat_C += 1
                qrs_c[Beat_C] = pks[i]
                qrs_i[Beat_C] = locs[i]

                if (y_i >= THR_SIG1):
                    Beat_C1 += 1
                    if ser_back:
                        qrs_i_raw[Beat_C1] = x_i
                    else:
                        qrs_i_raw[Beat_C1] = locs[i] - np.around(0.150*fs) + (x_i - 1)

                    qrs_amp_raw[Beat_C1] =  y_i
                    SIG_LEV1 = 0.125*y_i + 0.875*SIG_LEV1

                SIG_LEV = 0.125*pks[i] + 0.875*SIG_LEV

        elif ((THR_NOISE <= pks[i]) and (pks[i] < THR_SIG)):
            NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1
            NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV     
        elif (pks[i] < THR_NOISE):
            nois_c[Noise_Count] = pks[i]
            nois_i[Noise_Count] = locs[i]    
            NOISE_LEV1 = 0.125*y_i + 0.875*NOISE_LEV1    
            NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV
            Noise_Count += 1

        # Adjust the threshold with SNR
        if (NOISE_LEV != 0 or SIG_LEV != 0):
            THR_SIG = NOISE_LEV + 0.25*(abs(SIG_LEV - NOISE_LEV))
            THR_NOISE = 0.5*(THR_SIG)

        if (NOISE_LEV1 != 0 or SIG_LEV1 != 0):
            THR_SIG1 = NOISE_LEV1 + 0.25*(abs(SIG_LEV1 - NOISE_LEV1))
            THR_NOISE1 = 0.5*(THR_SIG1)

        SIGL_buf[i] = SIG_LEV
        NOISL_buf[i] = NOISE_LEV
        THRS_buf[i] = THR_SIG

        SIGL_buf1[i] = SIG_LEV1
        NOISL_buf1[i] = NOISE_LEV1
        THRS_buf1[i] = THR_SIG1

        skip = 0                                                  
        not_nois = 0
        ser_back = 0

    # Adjust lengths
    qrs_i_raw = qrs_i_raw[0:Beat_C1+1]
    qrs_amp_raw = qrs_amp_raw[0:Beat_C1+1]
    qrs_c = qrs_c[0:Beat_C+1]
    qrs_i = qrs_i[0:Beat_C+1]

    return qrs_i_raw

def qrs_detect(ECG, fs):
	winsize = 5 * fs * 60 # 5min 滑窗
	#winsize = 10 * fs # 10s 滑窗
	NB_SAMP = len(ECG)
	peaks = []
	if NB_SAMP < winsize:
		peaks.extend(p_t_qrs(ECG, fs))
		peaks = np.array(peaks)
		peaks = np.delete(peaks, np.where(peaks >= NB_SAMP-2*fs)[0]) # 删除最后2sR波位置
	else:
		# 5分钟滑窗检测，重叠5s数据
		count = NB_SAMP // winsize
		for j in range(count+1):
			if j == 0:
				ecg_data = ECG[j*winsize: (j+1)*winsize]

				peak = p_t_qrs(ecg_data, fs)
				peak = np.array(peak)
				peak = np.delete(peak, np.where(peak >= winsize-2*fs)[0]).tolist() # 删除5分钟窗口最后2sR波位置

				peaks.extend(map(lambda n: n+j*winsize, peak))
			elif j == count:
				ecg_data = ECG[j*winsize-5*fs: ]
				if len(ecg_data) == 0:
					pass
				else:
					peak = p_t_qrs(ecg_data, fs)
					peak = np.array(peak)
					peak = np.delete(peak, np.where(peak <= 2*fs)[0]).tolist() # 删除最后多余窗口前2sR波位置

					peaks.extend(map(lambda n: n+j*winsize-5*fs, peak))
			else:
				ecg_data = ECG[j*winsize-5*fs: (j+1)*winsize]
				peak = p_t_qrs(ecg_data, fs)
				peak = np.array(peak)
				peak = np.delete(peak, np.where((peak <= 2*fs) | (peak >= winsize-2*fs))[0]).tolist() # 删除中间片段5分钟窗口前2s和最后2sR波位置

				peaks.extend(map(lambda n: n+j*winsize-5*fs, peak))

	peaks = np.array(peaks)
	peaks = np.sort(peaks)
	dp = np.abs(np.diff(peaks))

	final_peaks = peaks[np.where(dp >= 0.2*fs)[0]+1]

	return final_peaks

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