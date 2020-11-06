import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def GetData():
    A = np.genfromtxt("C:\\Users\\kevin\\Documents\\Machine_Learning\\Machine_Learning\\Project 2\\MovieRankings36.csv", delimiter=",")
    om = np.argwhere(~np.isnan(A))#finding nan positions to be able to navigate around them later on
    omc = np.argwhere(np.isnan(A))
    return A, om, omc
def Omega_i(om, i):
    return np.array([j for (a,j) in om if a==i])
    
def Omega_j(om, j):
       return np.array([i for (i,a) in om if a==j])
    
def mask(A, inds):
    out = A
    for tup in inds:
        out[tup[0], tup[1]]=0
    return out
    
def ForceBound(M,n, d):
    minVal = np.ones(n,d)
    maxVal = 5 * minVal
    M = np.maximum(M,minVal)
    M = np.minimum(M,maxVal)
    return M
    
def LowRankFactorization(iterations, lam, x0, y0, om, omc, A, k):
    (n,d) = A.shape
    tol = 10**-3
    X = x0
    Y = y0
    for a in range(iterations):
        for i in range(n):
            om_i = Omega_i(om,i)
            yom_i = Y[om_i]
            yom_it = np.transpose(yom_i)
            X[i] = np.linalg.solve(yom_it@yom_i + lam*np.eye(k), yom_it@A[i,om_i])
        for j in range(d):
            om_j = Omega_j(om,j)
            xom_j = X[om_j]
            xom_jt = np.transpose(xom_j)
            Y[j] = np.linalg.solve(xom_jt@xom_j + lam*np.eye(k), xom_jt@A[om_j,j])
        M = X@np.transpose(Y)
        B = mask(A-M,omc)
        norm = np.linalg.norm(B, 'fro')
        if norm < tol:
            break
        return(X,Y)
def s_lam(M, lam):
    (U, S, Vt) = np.linalg.svd(M)
    l = len(S)
    U = U[:,:l]
    S = np.diag(np.maximum(S-lam*np.ones(l), np.zeros(l)))
    return (U@S)@Vt
def Nuclear_Iterations(A, om, omc, lam, iterations):
    M = np.zeros(A.shape)
    for q in range (iterations):
        M = s_lam(M + mask(A-M,omc), lam)
        #B = mask(A-M, omc)
        #norm = np.linalg.norm('B', 'fro')
        #print('iter ' + str(q) + '' + str(norm)+''+str(np.linalg.norm(M, 'fro')))
        if q%1000==0:
            print('running')
    return M
def k_means(A, k, iterations):
    (n,d)=A.shape
    inds = np.random.permutation(n)[:k]
    cluster_means = A[inds]
    labels = np.zeros(n)
    newlabels = np.zeros(n)
    
    for q in range(iterations):
        for i in range(n):
            distances = np.linalg.norm(A[i]-cluster_means, axis = 1)
            newlabels[i] = np.argmin(distances)
        if (newlabels == labels).all():
            break
        labels = newlabels.copy()
        
        for j in range(k):
            clust = A[np.argwhere(labels==j*np.ones(n)).flatten() ]
            cluster_means[j] = np.mean(clust, axis = 0)
            
    print ('completed in ' +str(q) + 'iterations')
    return (labels, cluster_means)
            
def projected_grad_descent(A, k, iterations,W,H):
    tol = 1
    (n,d) = A.shape
    R = A-(W@H)
    record = np.zeros(iterations+1)
    record[0] = np.linalg.norm(R,'fro')
    
    for i in range(iterations):
        a = Step_Size(A, W, H)
        W_new = np.maximum(W+a*R@np.transpose(H), 0)
        H_new = np.maximum(H+a*np.transpose(W)@R, 0)
        
        W = W_new.copy()
        H = H_new.copy()
        R = A-(W@H)
        record[i+1] = np.linalg.norm(R, 'fro')
    
        if record[i+1]<tol:
            break
    record[i+2:] = record[i+1]*np.ones(len(record[i+2:]))

    return(W,H,record)

def Step_Size(A, W, H):
    R0 =A - (W@H)
    n0 = (np.linalg.norm(R0, 'fro')**2)/2
    a=1
    Wgrad = R0@np.transpose(H)
    
    Hgrad = np.transpose(W)@R0
    
    ng = np.sqrt(np.linalg.norm(Hgrad, 'fro')**2+np.linalg.norm(Wgrad,'fro')**2)
    r = 0.9
    for i in range(200):
        W_new = np.maximum(W+a*R0@np.transpose(H), 0)
        H_new = np.maximum(H+a*np.transpose(W)@R0, 0)
        R = A - W_new@H_new
        n = (np.linalg.norm(R, 'fro')**2)/2
        if n<n0-0.5*a*ng:
            break
        else: a = a*r
    return a 
    
def Lee_Seung(A, k, iterations, W, H):
    tol = 1
    (n,d) = A.shape
    record = np.zeros(iterations+1)
    R = A-(W@H)
    record[0] = np.linalg.norm(R,'fro')

    for i in range(iterations):
        Wt = np.transpose(W)
        H = H*(Wt@A)/(Wt@W@H)
        Ht = np.transpose(H)
        W = W*(A@Ht)/(W@H@Ht)
        R = A - (W@H)
        record[i+1] = np.linalg.norm(R, 'fro')
    
        if record[i+1]<tol:
            break
    record[i+2:] = record[i+1]*np.ones(len(record[i+2:]))

    return(W,H,record)

def Get_CUR_Data():
    M = np.genfromtxt("C:\\Users\\kevin\\Documents\\Machine_Learning\\Machine_Learning\\Project 2\\M1.csv", delimiter=",")
    y = np.genfromtxt("C:\\Users\\kevin\\Documents\\Machine_Learning\\Machine_Learning\\Project 2\\y1.csv", delimiter=",")
    return M,y
def Column_Select(M, k, c):
    (U, S, Vt) = np.linalg.svd(M)
    (n,d)=M.shape
    Vtk = Vt[:k,:]
    lev = np.mean(Vtk*Vtk, axis=0)
    r = np.random.rand(d)
    cols = np.argwhere(r<c*lev).flatten()
    return M[:,cols]
    
def CUR_decomp(M, y, k, a):
   c = a*k
   C = Column_Select(M, k, c)
   R = np.transpose(Column_Select(np.transpose(M), k, c))
   U = np.linalg.pinv(C)@M@np.linalg.pinv(R) 
   return(C,R,U)

#def Select_Data_4():
    
def Prob_4(M):
    k = 10
    a = 8
    k = a*k
   # M = M[:,0:10000]
    M1 = M[0:71,:]
    M2 = M[71:139,:]
    
    (S,U,Vt) = np.linalg.svd(M)
    (n,d) = M.shape
    lev_c1 = np.zeros(n)
    lev_c2 = np.zeros(n)
    #computing leverage of column vectors of M1
    for j in range(d):
       lev_c1[j] = (1/k)*np.sum(M1[:,j]**2)
    #computing leverage of column vectors of M2
    for j in range(d):
       lev_c2[j] = (1/k)*np.sum(M2[:,j]**2)
    w = lev_c1-lev_c2
    w1 = np.sort(w)
    w2 = w1[:,k]
    #making matrix 
    
    
    Vt = Vt[:,0:2]
    M = M@Vt
    
    return M
        
    
