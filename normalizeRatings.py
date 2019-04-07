import numpy as np
def Func(Y, R):
    m, n = np.shape(Y)
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros((m, n))
    for i in range(m):
        idx = np.nonzero(R[i, :] == 1)[0]
        Ymean[i] = Y[i, idx].mean(axis = 0)
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ymean, Ynorm