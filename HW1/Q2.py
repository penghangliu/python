
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# In[2]:


file = 'bipartite.txt'
f = open(file, 'r') 
lines = f.readlines()
G = nx.Graph()
left = []
right = []
for i in range(2,len(lines)):
    p = lines[i].split()
    left.append(int(p[0]))
    right.append(int(p[1]))
f.close()
left = np.array(left)
right = np.array(right)


# In[3]:


df = pd.DataFrame(0, range(1,left.max()+1), range(1,right.max()+1))
for i in range(left.shape[0]):
    df[right[i]][left[i]] += 1


# In[4]:


G1 = df.dot(df.transpose())
G2 = df.transpose().dot(df)

G1[G1 != 0] = 1
G1.values[[np.arange(G1.shape[0])]*2] = 0
G2[G2 != 0] = 1
G2.values[[np.arange(G2.shape[0])]*2] = 0


# In[5]:


deg_left = np.array(G1.sum())[G1.sum().nonzero()]
print('Left nodes:',np.count_nonzero(deg_left))
print('Left edges:',int(deg_left.sum()/2))


# In[6]:


deg_right = np.array(G2.sum())[G2.sum().nonzero()]
print('Right nodes:',np.count_nonzero(deg_right))
print('Right edges:',int(deg_right.sum()/2))


# In[7]:


fig, ax =  plt.subplots(figsize=(10,7))
y, x, _ = ax.hist(deg_left,bins=15,rwidth=0.7)

ax.set_ylabel('Number of Nodes \n' + 'Min: ' + str(int(y.min())) + ' Max: ' + str(int(y.max())), fontsize=20)
ax.set_xlabel('Degree \n' + 'Min: ' + str(int(x.min())) + ' Max: ' + str(int(x.max())), fontsize=20)
fig.suptitle('Left Projection Degreee distribution',fontsize =30)
plt.savefig('Left Degree distribution.png')


# In[8]:


fig, ax =  plt.subplots(figsize=(10,7))
y, x, _ = ax.hist(deg_right,bins=15,rwidth=0.7)

ax.set_ylabel('Number of Nodes \n' + 'Min: ' + str(int(y.min())) + ' Max: ' + str(int(y.max())), fontsize=20)
ax.set_xlabel('Degree \n' + 'Min: ' + str(int(x.min())) + ' Max: ' + str(int(x.max())), fontsize=20)
fig.suptitle('Right Projection Degreee distribution',fontsize =30)
plt.savefig('Right Degree distribution.png')

