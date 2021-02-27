from sklearn import preprocessing
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt
import scipy.io
import numpy as np
import pandas as pd


def pan_tomken(data, fs):
    N = 24

    rE = fs//3
    E = fs//7

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
    c=0
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
    Q = np.zeros(2)
    S = np.zeros(2)
    THR = 50
    for i in peak:
        if(i==0 or i<2*rE):
            continue
        p[i]=1

        ind = np.argmax(x2[i-rE:i+rE])
        maxIndexR = (ind+i-rE)
        rPeak.append(maxIndexR)
        plt.plot(maxIndexR,x2[maxIndexR],'ro', markersize=12)
        prevDiffQ = 0
        prevDiffS = 0

    #    FIND THE Q POINT
        for i in range(1,THR):

            Q[0] = x2[maxIndexR-i]
            Q[1] = x2[maxIndexR-(i+1)]

            diffQ = Q[0]-Q[1]

            if(diffQ<prevDiffQ):
                minIndexQ = maxIndexR-i
                break
            prevDiffQ = diffQ / 5

    #    FIND THE S POINT
        for i in range(1,THR):

            S[0] = x2[maxIndexR+i]
            S[1] = x2[maxIndexR+(i+1)]

            diffS = S[0]-S[1]

            if(diffS<prevDiffS):
                minIndexS = maxIndexR+i
                break
            prevDiffS = diffS / 5

    rPeak = np.unique(rPeak)

    return rPeak