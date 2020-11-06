"""In this file we load in the completed matrix from probelm 2. Then we call our functions to cluster the data
and to compute the NMF of our matrix. We compute the NMF first with the Projected Gradient Descent method
then with the Lee-Seung method and finally we begin with the PGD and then finish with the Lee-Seung method.
We then plot the Frobenius norm of R vs the iteration number to see how quickly it decays for each algorithm.  """
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import Methods as meth
import pickle
#obtaining data
pickle_in = open("complete_matrix.p", "rb")
A = pickle.load(pickle_in)
(n,d) = A.shape
k = 12
W0 = 2*np.random.random((n,k))/np.sqrt(k/3)
H0 = 2*np.random.random((k,d))/np.sqrt(k/3)
X0 = np.zeros((n,k))
Y0 = np.zeros((k,d))
"""
#Clustering Data
(labels,cluster_means) = meth.k_means(A,k, 1000)

np.savetxt("clustered.csv", labels, delimiter=",")
np.savetxt("raw.csv", A, delimiter=",")
"""

#finding NMF of data using PGD
(W,H,f) = meth.projected_grad_descent(A, k, 1000, W0, H0)

#Finding the NMF using Lee-Seung
(W1, H1, f1) = meth.Lee_Seung(A, k, 1000, W0, H0)

#Finding the NMF of data using PGD then Lee-Seung
(W3,H3,f3) = meth.projected_grad_descent(A, k, 200, W0, H0)

(W4, H4, f4) = meth.Lee_Seung(A, k, 800, W3, H3)

f3 = f3[0:200]
print(f3.shape)
f4 = np.append(f3,f4,0)

x = (np.arange(len(f)))

f = np.log(f)
f1 = np.log(f1)
f4 = np.log(f4)


fig = plt.figure()
#fig.subplots_adjust(top=0.8)
ax = fig.add_subplot(111)
plt.xscale('log')
lab = 'PGD'
plt.plot(x, f, label = lab) 
lab = 'Lee-Seung'
plt.plot(x, f1, label = lab) 
lab = 'PGD & Lee-Seung'
plt.plot(x, f4, label = lab) 
ax.set_xlabel('iterations number')
ax.set_ylabel('frobinius norm')
plt.legend()
plt.show()




