
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# In[86]:


file = 'directed.txt'
f = open(file, 'r') 
lines = f.readlines()
s = []
t = []
for i in range(2,len(lines)):
    p = lines[i].split()
    s.append(int(p[0]))
    t.append(int(p[1]))
f.close()
s = np.array(s)
t = np.array(t)


# In[168]:


M = pd.DataFrame(0, range(1,max(s.max(),t.max())+1), range(1,max(s.max(),t.max())+1))
M1 = pd.DataFrame(0, range(1,max(s.max(),t.max())+1), range(1,max(s.max(),t.max())+1))
M2 = pd.DataFrame(0, range(1,max(s.max(),t.max())+1), range(1,max(s.max(),t.max())+1))
for i in range(s.shape[0]):
    M[s[i]][t[i]] += 1

for i in range(1,M1.shape[0]+1):
    M1[i]=M[i]/M[i].sum()
    
for i in range(1,M2.shape[0]+1):
    if (M[i].sum()!=1):
        M2[i]=M[i]/(M[i].sum() ** 2 - M[i].sum())
    else:
        M2[i]=0


# In[217]:


def PR(M,beta,iteration):
    n = M.shape[0]
    s = np.full(n,1/n)
    for i in range(iteration):
        a = beta * np.dot(M,s)
        T = a.sum()
        s = a + (1-T)/n
    return s


# In[246]:


s1 = PR(M1,0.75,50)
s2 = PR(M2,0.75,50)


# In[245]:


out1 = pd.DataFrame({'ID': np.arange(500)+1,'PageRank':s1})
print(out1.nlargest(10, 'PageRank'))


# In[247]:


out2 = pd.DataFrame({'ID': np.arange(500)+1,'PageRank':s2})
print(out2.nlargest(10, 'PageRank'))

