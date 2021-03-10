import json
import numpy as np
import sys

import matplotlib.pyplot as plt
import pandas as pd
import peakutils
from sklearn import preprocessing
import scipy.signal as ss
from scipy import io
from scipy.signal import (
	butter,filtfilt,
	medfilt,iirnotch,
	convolve
)

def qrs_detect(ECG, fs):
	winsize = 5 * fs * 60 # 5min 滑窗
	#winsize = 10 * fs # 10s 滑窗
	NB_SAMP = len(ECG)
	peaks = []
	if NB_SAMP < winsize:
		ecg_prep = preprocessing(fs).run(ECG)
		peaks.extend(qrs_detection(fs, REF_PERIOD=0.22, THRES=0.08, debug=0).run(ecg_prep))
		peaks = np.array(peaks)
		peaks = np.delete(peaks, np.where(peaks >= NB_SAMP-2*fs)[0]) # 删除最后2sR波位置
	else:
		# 5分钟滑窗检测，重叠5s数据
		count = NB_SAMP // winsize
		for j in range(count+1):
			if j == 0:
				ecg_data = ECG[j*winsize: (j+1)*winsize]
				ecg_prep = preprocessing(fs).run(ecg_data)

				peak = qrs_detection(fs, REF_PERIOD=0.22, THRES=0.08, debug = 0).run(ecg_prep)
				peak = np.array(peak)
				peak = np.delete(peak, np.where(peak >= winsize-2*fs)[0]).tolist() # 删除5分钟窗口最后2sR波位置

				peaks.extend(map(lambda n: n+j*winsize, peak))
			elif j == count:
				ecg_data = ECG[j*winsize-5*fs: ]
				if len(ecg_data) == 0:
					pass
				else:
					ecg_prep = preprocessing(fs).run(ecg_data)

					peak = qrs_detection(fs, REF_PERIOD=0.22, THRES=0.08, debug = 0).run(ecg_prep)
					peak = np.array(peak)
					peak = np.delete(peak, np.where(peak <= 2*fs)[0]).tolist() # 删除最后多余窗口前2sR波位置

					peaks.extend(map(lambda n: n+j*winsize-5*fs, peak))
			else:
				ecg_data = ECG[j*winsize-5*fs: (j+1)*winsize]
				ecg_prep = preprocessing(fs).run(ecg_data)

				peak = qrs_detection(fs, REF_PERIOD=0.22, THRES=0.08, debug = 0).run(ecg_prep)
				peak = np.array(peak)
				peak = np.delete(peak, np.where((peak <= 2*fs) | (peak >= winsize-2*fs))[0]).tolist() # 删除中间片段5分钟窗口前2s和最后2sR波位置

				peaks.extend(map(lambda n: n+j*winsize-5*fs, peak))

	peaks = np.array(peaks)
	peaks = np.sort(peaks)
	dp = np.abs(np.diff(peaks))

	final_peaks = peaks[np.where(dp >= 0.2*fs)[0]+1]

	return final_peaks

class preprocessing(object):

	def __init__(self, fs):
		self.highpass = 15
		self.swin = 30
		self.lwin = 300

		self.fs = fs
		self.nyq = fs * 0.5
		return

	def __signal_reshape(self, signal):
		signal = signal.reshape((-1, 1))

		return signal, len(signal)

	def __norm(self, signal):
		normed_signal = signal / np.max(np.abs(signal))

		return normed_signal

	def __enhancement(self, signal):
		signal = self.__remove_outliers(signal, 3)
		signal2 = np.power(signal, 2)

		swin_filter = np.ones((self.swin, 1))
		lwin_filter = np.ones((self.lwin, 1))

		signal2_sf = convolve(signal2, swin_filter, 'same', 'auto')
		signal2_lf = convolve(signal2, lwin_filter, 'same', 'auto')

		coeff = signal2_sf / (signal2_lf + 1e-3)
		enhanced_signal = coeff * signal
		enhanced_signal = self.__norm(enhanced_signal)

		return enhanced_signal

	def __remove_outliers(self, signal, t):
		signalc = np.copy(signal)
		std = np.std(signalc)
		t_std = t * std
		outliers = np.where(np.abs(signalc) > t_std)
		signalc[outliers] = t_std
		signalc = self.__norm(signalc)

		return signalc

	def __highpass_filter(self, signal, hpf):
		hp = hpf / self.nyq
		hpb, hpa = butter(5, hp, 'highpass')
		hp_signal = filtfilt(hpb, hpa, signal, axis=0)
		hp_signal = self.__norm(hp_signal)

		return hp_signal

	def __notch_filter(self, signal, ntf):
		nt = ntf / self.nyq
		ntb, nta = iirnotch(nt, 30)
		nt_signal = filtfilt(ntb, nta, signal, axis=0)
		nt_signal = self.__norm(nt_signal)

		return nt_signal

	def __median_filter(self, signal, size=5):
		med_signal = medfilt(signal.flatten(), size).reshape((-1, 1))
		med_signal = self.__norm(med_signal)
		return med_signal

	def run(self, signal):
		# signal = self.__signal_reshape(signal)
		md_signal = self.__median_filter(signal, 5)

		fp_signal = self.__highpass_filter(md_signal, self.highpass)
		fp_signal = self.__notch_filter(fp_signal, 50)
		fp_signal = self.__enhancement(fp_signal)

		prep_signal = fp_signal[:, -1]

		return prep_signal


class qrs_detection(object):

	def __init__(self, fs, debug, REF_PERIOD, THRES):
		self.REF_PERIOD = REF_PERIOD # 0.28  0.22
		self.THRES = THRES  # 0.47  0.08
		self.fs = fs
		self.debug = debug
		self.SEARCH_BACK = 1
		self.MAX_FORCE = []
		self.MED_SMOOTH_NB_COEFF = round(self.fs / 100)
		self.INT_NB_COEFF = int(round(30 * self.fs / 256))
		self.b1 = [-7.76134153004032e-05, -8.61340065700186e-07, -0.000235896265227759, -0.000218187348080493,
		           -0.000669276661989076, -0.000895315769128474, -0.00177103523376593, -0.00261090269427532,
		           -0.00436658562236147, -0.00654042051038359, -0.0100184331404942, -0.0147404490576768, -0.0213552862626397,
		           -0.0304383713068943, -0.0422046489494271, -0.0579080696611326, -0.0771207767471077, -0.101522164227050,
		           -0.129807529156737, -0.163496603270115, -0.200168449008964, -0.240386155222128, -0.280473635134792, -0.319366250886081,
		           -0.352321273862402, -0.376564022211588,  -0.387067857380138, -0.380212957685603, -0.352156166419805, -0.300190214010979,
		           -0.223478907410178, -0.122231287760963, 1.10369133101340e-19, 0.138912393694904, 0.286575666377766, 0.435190138017945,
		           0.574355824775253, 0.695302079844553, 0.788508084499882, 0.847746151348478, 0.867773885805332, 0.847746151348478,
		           0.788508084499882, 0.695302079844553, 0.574355824775253, 0.435190138017945, 0.286575666377766, 0.138912393694904,
		           1.10369133101340e-19, -0.122231287760963, -0.223478907410178, -0.300190214010979, -0.352156166419805, -0.380212957685603,
		           -0.387067857380138, -0.376564022211588, -0.352321273862402, -0.319366250886081, -0.280473635134792, -0.240386155222128,
		           -0.200168449008964, -0.163496603270115, -0.129807529156737, -0.101522164227050, -0.0771207767471077, -0.0579080696611326,
		           -0.0422046489494271, -0.0304383713068943, -0.0213552862626397, -0.0147404490576768, -0.0100184331404942, -0.00654042051038359,
		           -0.00436658562236147, -0.00261090269427532, -0.00177103523376593, -0.000895315769128474, -0.000669276661989076, -0.000218187348080493,
		           -0.000235896265227759, -8.61340065700213e-07, -7.76134153004032e-05, 6.77368189249316e-05]
		return

	def __gqrs(self, ecg):
		NB_SAMP = np.size(ecg)
		tm = np.arange(1 / self.fs, np.ceil(NB_SAMP / self.fs), 1 / self.fs).reshape((-1, 1))
		#tm = np.expand_dims(np.arange(1 / self.fs, np.ceil(NB_SAMP / self.fs), 1 / self.fs), -1)
		maxval = []
		maxloc = []
		qrs_pre = []

		bandpass_signal = ss.filtfilt(self.b1, 1, ecg)
		bpfecg = bandpass_signal[0:NB_SAMP]

		dffecg = np.diff(bpfecg)
		sqrecg = np.power(dffecg, 2)
		sqrecg = sqrecg.reshape((-1, 1))[:, -1]
		intecg = ss.lfilter(np.ones(self.INT_NB_COEFF), np.ones(1), sqrecg)
		mdfint = ss.medfilt(intecg, 5)
		delay = np.ceil(self.INT_NB_COEFF / 2)
		delay = int(delay)
		mdfint = np.roll(mdfint, -delay)
		mdfint = mdfint / np.max(np.abs(mdfint))

		if self.debug:
			plt.figure(figsize=(60, 12))
			plt.plot(ecg)
			plt.plot(mdfint)
			plt.title('Prep + normalization')
			plt.show()
		if NB_SAMP / self.fs > 90:
			xs = np.sort(mdfint[self.fs:self.fs * 90])
		else:
			xs = np.sort(mdfint[self.fs:-1])

		if len(self.MAX_FORCE) == 0:
			if NB_SAMP / self.fs > 10:
				ind_xs = int(np.ceil(98 / 100 * len(xs)))
				en_thres = xs[ind_xs]
			else:
				ind_xs = int(np.ceil(99 / 100 * len(xs)))
				en_thres = xs[ind_xs]
		else:
			en_thres = self.MAX_FORCE
		poss_reg = mdfint > (self.THRES * en_thres)
		if len(poss_reg) == 0:
			poss_reg[9] = 1

		if self.SEARCH_BACK:
			indAboveThreshold = np.where(poss_reg)[0]
			RRv = np.diff(tm[indAboveThreshold][:, -1])
			medRRv = np.median(RRv[np.where(RRv > 0.01)[0]])
			indMissedBeat = np.where(RRv > 1.5 * medRRv)[0]
			indStart = indAboveThreshold[indMissedBeat]
			indEnd = indAboveThreshold[indMissedBeat + 1]

			for i in range(len(indStart)):
				poss_reg[indStart[i]: indEnd[i]] = mdfint[indStart[i]: indEnd[i]] > (0.5 * self.THRES * en_thres)
		poss_reg = poss_reg.astype(int)
		left = np.where(np.diff(np.pad(poss_reg, ((1, 0)), 'constant')) == 1)[0]
		right = np.where(np.diff(np.pad(poss_reg, ((0, 1)), 'constant')) == -1)[0]

		nbs = int(len(left < 30 * self.fs))
		for i in range(nbs):
			if left[i] == right[i]:
				right[i] += 1

		loc = np.zeros(nbs, dtype=int)
		for j in range(nbs):
			loc[j] = np.argmax(np.abs(bpfecg[left[j]: right[j]]))
			loc[j] = int(loc[j] - 1 + left[j])
		sign = np.mean(bpfecg[loc])

		for i in range(len(left)):
			if sign > 0:
				maxval.append(np.max(bpfecg[left[i]:right[i]]))
				maxloc.append(np.argmax(bpfecg[left[i]:right[i]]))
			else:
				maxval.append(np.min(bpfecg[left[i]:right[i]]))
				maxloc.append(np.argmin(bpfecg[left[i]:right[i]]))
			maxloc_int = list(map(int, maxloc))
			qrs_pre.append(maxloc_int[i] - 1 + left[i])

			if i > 0:
				if (qrs_pre[-1] - qrs_pre[-2] < self.fs * self.REF_PERIOD) & (np.abs(maxval[-1]) >= np.abs(maxval[-2])):
					del qrs_pre[-2]
				elif (qrs_pre[-1] - qrs_pre[-2] < self.fs * self.REF_PERIOD) & (
						np.abs(maxval[-1]) < np.abs(maxval[-2])):
					qrs_pre.pop()

		qrs_pos = qrs_pre
		if self.debug:
			plt.figure(figsize=(60, 12))
			plt.plot(ecg)
			plt.plot(qrs_pre,ecg[qrs_pos],'ro')
			plt.show()

		return qrs_pos

	def run(self, signal):

		qrs_pos = self.__gqrs(signal)

		return qrs_pos

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