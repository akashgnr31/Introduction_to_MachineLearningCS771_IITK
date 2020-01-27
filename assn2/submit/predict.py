import numpy as np
from numpy import random as rand
import subprocess
from _svmlight_loader import _dump_svmlight_file
from scipy import sparse as sps
import time as tm

def preprocess(N,d):
    f = open("new.X","r")
    fr = f.readlines()
    fw = open("newX.X","w+")
    fw.write(str(N)+" "+str(d)+"\n")
    for i in range(len(fr)):
        fw.write(fr[i][2:-2]+"\n")
        
def dump_svmlight_file(X, y, f, zero_based=True):
    if hasattr(f, "write"):
        raise ValueError("File handler not supported. Use a file path.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X.shape[0] and y.shape[0] should be the same, "
                         "got: %r and %r instead." % (X.shape[0], y.shape[0]))
    X = sps.csr_matrix(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    _dump_svmlight_file(f, X.data, X.indices, X.indptr, y, int(zero_based))


def getReco( X, k ):
    N, d = X.get_shape()
    dummy = sps.csr_matrix( (N, 1) )
    y = np.ones((N))
    dump_svmlight_file(X, y, "new.X", zero_based=True)
    preprocess(N,d)
    subprocess.call("./fastXML_predict newX.X score.txt model", shell=True)
    f = open("score.txt","r")
    fr = f.readlines()
    yPred = np.zeros((N,k))
    for i in range(len(fr)):
        if i==0:
            continue
        a = fr[i].split(" ")
        tmp1 = np.zeros(len(a))
        tmp2 = np.zeros(len(a))
        for j in range(len(a)):
            b = np.array(a[j].split(":"))
            tmp1[j] = b[0].astype(int)
            tmp2[j] = b[1].astype(float)
        label_ind = np.argsort(-tmp2)[:k]
        yPred[i-1] = tmp1[label_ind]

    return yPred
