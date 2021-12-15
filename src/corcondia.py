'''
source: https://gist.github.com/willshiao/2c0d7cc1133d8fa31587e541fef480fb
Based on the Python implementation by Alessandro Bessi at https://github.com/alessandrobessi/corcondia
  and the original MATLAB implementation by Evangelos (Vagelis) Papalexakis at https://www.cs.ucr.edu/~epapalex/src/efficient_corcondia.zip
Updated to work with modern versions of Python.
Uses Tensorly, and assumes that we use the numpy backend.
References:
Buis, Paul E., and Wayne R. Dyksen. 
"Efficient vector and parallel manipulation of tensor products." 
ACM Transactions on Mathematical Software (TOMS) 22.1 (1996): 18-23.
Papalexakis, Evangelos E., and Christos Faloutsos. 
"Fast efficient and scalable core consistency diagnostic 
for the parafac decomposition for big sparse tensors." 
2015 IEEE International Conference on Acoustics, 
Speech and Signal Processing (ICASSP). IEEE, 2015.
Bro, Rasmus, and Henk AL Kiers. 
"A new efficient method for determining the number of components in PARAFAC models."
Journal of chemometrics 17.5 (2003): 274-286.
'''

import tensorly as tl
from tensorly.tenalg import mode_dot
from tensorly.decomposition import parafac
import numpy as np

def kronecker_mat_ten(matrices, X):
    for k in range(len(matrices)):
        M = matrices[k]
        Y = mode_dot(X, M, k)
        X = Y
        X = tl.moveaxis(X, [0, 1, 2], [2, 1, 0])
    return Y

def corcondia(X, k = 1, init='random', **kwargs):
    weights, X_approx_ks = parafac(X, k, init=init, **kwargs)

    A, B, C = X_approx_ks
    x = tl.cp_to_tensor((weights, X_approx_ks))

    Ua, Sa, Va = np.linalg.svd(A)
    Ub, Sb, Vb = np.linalg.svd(B)
    Uc, Sc, Vc = np.linalg.svd(C)

    SaI = np.zeros((Ua.shape[0], Va.shape[0]), float)
    np.fill_diagonal(SaI, Sa)

    SbI = np.zeros((Ub.shape[0], Vb.shape[0]), float)
    np.fill_diagonal(SbI, Sb)

    ScI = np.zeros((Uc.shape[0], Vc.shape[0]), float)
    np.fill_diagonal(ScI, Sc)

    SaI = np.linalg.pinv(SaI)
    SbI = np.linalg.pinv(SbI)
    ScI = np.linalg.pinv(ScI)

    part1 = kronecker_mat_ten([Ua.T, Ub.T, Uc.T], x)
    part2 = kronecker_mat_ten([SaI, SbI, ScI], part1)
    G = kronecker_mat_ten([Va.T, Vb.T, Vc.T], part2)

    T = np.zeros((k, k, k))
    for i in range(k):
        T[i,i,i] = 1

    return 100 * (1 - ((G-T)**2).sum() / float(k))