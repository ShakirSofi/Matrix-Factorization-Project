"""In this file we complete the matrix by computing the Low rank facorization of the matrix. We then do implement the 
nuclear norm trick. The resulting matrix is then saved at the end for later use"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import Methods as meth
import pickle

(A,om,omc) = meth.GetData()
(n,d) = A.shape
k = 10
X0 = np.ones((n,k))
Y0 = np.ones((d,k))
(X,Y) = meth.LowRankFactorization(100,1, X0, Y0, om, omc, A, k)
A1 = X@np.transpose(Y)

print(A1)
pickle_out = open("dict.pickle","wb")
pickle.dump(A1,pickle_out)
pickle_out.close()
