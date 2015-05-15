import numpy as np
import random as rd
adj = [1.0,1.0,1.0,1.0]
#adj = [1.0,3.0,1.0]
Lh = 5
def ini(x,y):
    m = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            m[i][j] = rd.gauss(0,1)
    return m
def forward(lan):
    Lx = len(lan[0])
    U = ini(Lh,Lx)
    V = ini(Lx,Lh)
    W = ini(Lh,Lh)
    s = [0]*(len(lan)+1)
    s[0] = [0]*Lh
    s[-1] = [0]*Lh
    rate =1
    for p in range(10000):
        for i in range(len(lan)-1):
            z = np.array(np.dot(U,lan[i])+np.dot(W,s[i]))
            s[i+1] = 1/(np.exp(z*-1)+1)
            y = np.exp(np.dot(V,s[i+1]))
            y = y/np.sum(y)
            err = np.array(y-lan[i+1])
            de = np.dot(np.transpose(V),y*err)
            gv = np.outer(y*err,s[i+1])
            gu = np.outer(de,lan[i])
            gw = np.outer(de,s[i])
            W = W-gw*rate*adj[i]
            V = V-gv*rate*adj[i]
            U = U-gu*rate*adj[i]
        s[i+1] = s[i]
        rate *=0.9999
    print(y)
    print("=======")
    for i in range(len(lan)):
        z = np.array(np.dot(U,lan[i])+np.dot(W,s[i]))
        s[i+1] = 1/(np.exp(z*-1)+1)
        y = np.exp(np.dot(V,s[i+1]))
        y = y/np.sum(y)
        print(y)

forward([[0.4,0.1,0.5],[0.5,0.1,0.4],[0.4,0.5,0.1],[0.1,0.4,0.5],[0.5,0.4,0.1]])
