from math import*
import numpy as np
import time
from random import*
import matplotlib.pyplot as pp
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

#Fonctions du TP précédent

def res_low(Aaug):
    n,m = np.shape(Aaug)
    Y = np.zeros(n)
    for i in range(0,n):
        somme = 0
        for k in range(0,n):
            somme = somme + Y[k]*Aaug[i,k]
        Y[i] = (Aaug[i,n]-somme)/Aaug[i,i]
    return Y

def res(Aaug):
    n,m = np.shape(Aaug)
    X = np.zeros(n)
    for i in range(n-1,-1,-1):
        somme = 0
        for k in range(i,n):
            somme = somme + X[k]*Aaug[i,k]
        X[i] = (Aaug[i,n]-somme)/Aaug[i,i]
    return X

#Question 1

def Trouver_A(dim):
    M = np.random.random_sample((dim,dim))
    if round(np.linalg.det(M),3)!=0:
        return np.dot(M.T,M)
    else:
        Trouver_A(dim)
        
def Cholesky(A):
    n,c = np.shape(A)
    L = np.zeros((n,c))
    
    for k in range(0,n):
        somme = 0
        for j in range(0,k):
            somme = somme+L[k,j]**2
    L[k,k] = sqrt(A[k,k]-somme)
    for i in range(k+1,n):
        somme = 0
        for j in range(0,k):
            somme = somme+L[i,j]*L[k,j]
        L[i,k] = (A[i,k]-somme)/L[k,k]
    LT = L.T
    return(L,LT)
        
#Question 2

def ResolCholesky(A,B):
    [L,LT] = Cholesky(A)
    Aaug = np.concatenate((L,B),axis = 1)
    n,m = np.shape(Aaug)
    
    Y = res_low(Aaug)
    Y = np.reshape(res_low(Aaug),(n,1))
    Aaug = np.concatenate((LT,Y),axis = 1)
    X = res(Aaug)
    
    return(X)

temps4=[]
indices4=[]
normes4=[]
for n in range (0,100,2):
    print(n)
    try :
        A = Trouver_A(n)
        B = np.random.random_sample((n,1))
        t1 = time.time()
        x = ResolCholesky(A,B)
        t2 = time.time()
        t = t2 - t1
        temps4.append(t)
        indices4.append(n)
        norme = np.linalg.norm(np.dot(A,x)-np.ravel())
        normes4.append(norme)
        print(x)
    except :
        print('')
       

df = pd.DataFrame(temps4)
df.to_csv('temps4.csv',index=False)

df2 = pd.DataFrame(indices4)
df2.to_csv('graphe4.csv',index=False)

df3 = pd.DataFrame(normes4)
df3.to_csv('normes4.csv',index=False)

temps=[]

indices=[]

normes=[]

dim=2000

 

 

 


for n in range (0,2000,2):

    print(n)

    try :

         A = Trouver_A(n)

         B = np.random.randint(low=0,high=n,size=(n,1))

         t1 = time.time()

         x = np.linalg.solve(A,B)

         print(x)

         t2 = time.time()

         t = t2 - t1

         temps.append(t)

         indices.append(n)

         norme = np.linalg.norm(A@x-B)

         normes.append(norme)

 

       

    except :

         print('')

      

 

df = pd.DataFrame(temps)

df.to_csv('temps.csv',index=False)

 

df2 = pd.DataFrame(indices)

df2.to_csv('indices.csv',index=False)

 

df3 = pd.DataFrame(normes)

df3.to_csv('normes.csv',index=False)

 

print("les fichiers excels sont prêts")

 



 

 



for n in range (0,2000,2):

    print(n)

    try :

         A = Trouver_A(n)

         B = np.random.randint(low=0,high=n,size=(n,1))

         t1 = time.time()

         L = np.linalg.cholesky(A)

         y = np.linalg.solve(L,B)

         x = np.linalg.solve(L.T,B)

         print(x)

         t2 = time.time()

         t = t2 - t1

         temps.append(t)

         indices.append(n)

         norme = np.linalg.norm(A@x-B)

         normes.append(norme)

 

       

    except :

         print('')

      

 

df = pd.DataFrame(temps)

df.to_csv('temps.csv',index=False)

 

df2 = pd.DataFrame(indices)

df2.to_csv('indices.csv',index=False)

 

df3 = pd.DataFrame(normes)

df3.to_csv('normes.csv',index=False)

 

print("les fichiers excels sont prêts")




for n in range (0,2000,2):

    print(n)

    try :

         A = Trouver_A(n)

         B = np.random.randint(low=0,high=n,size=(n,1))

         t1 = time.time()

         L, low = cho_factor(A)

         x = cho_solve((L,low),B)

 

         print(x)

         t2 = time.time()

         t = t2 - t1

         temps.append(t)

         indices.append(n)

         norme = np.linalg.norm(A@x-B)

         normes.append(norme)

 

       

    except :

         print('')

      

 

df = pd.DataFrame(temps)

df.to_csv('temps.csv',index=False)

 

df2 = pd.DataFrame(indices)

df2.to_csv('indices.csv',index=False)

 

df3 = pd.DataFrame(normes)

df3.to_csv('normes.csv',index=False)

 

print("les fichiers excels sont prêts")
