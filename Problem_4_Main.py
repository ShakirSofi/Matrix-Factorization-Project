import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import Methods as meth


one = np.zeros(100)


(M, y) = meth.Get_CUR_Data()

two = np.zeros((4,5))
q = 1
j = 2

for q in range(8):
    for j in range(10):
        for i in range(100):
            (C, R, U) = meth.CUR_decomp(M, y, j, q)
            one[i] = np.linalg.norm((M-C@U@R), 'fro')/np.linalg.norm((M-C), 'fro')
        two[q,j] = np.sum(one)/len(one)
        j = j +2
    q = q+2


x = np.array[2,4,6,8,10]
fig = plt.figure()
#fig.subplots_adjust(top=0.8)
ax = fig.add_subplot(111)
lab = 'a = 1'
plt.plot(x, two[0,:], label = lab) 
lab = 'a = 3'
plt.plot(x, two[1,:], label = lab) 
lab = 'a = 5'
plt.plot(x, two[2,:], label = lab) 
lab = 'a = 7'
plt.plot(x, two[3,:], label = lab) 
ax.set_xlabel('k')
ax.set_ylabel('mean ratio')
plt.legend()
plt.show()


