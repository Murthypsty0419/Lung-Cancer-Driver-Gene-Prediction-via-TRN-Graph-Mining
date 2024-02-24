#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import requirement libraries
import matplotlib.pyplot as plt
import networkx as nx
import collections
from igraph import *
import pandas as pd
import numpy as np
import scipy.stats
import operator
from matplotlib.pyplot import figure
import igraph
import community   ## for louvain algorithm
from sklearn.metrics import *


# In[2]:


df = pd.read_csv("WLUSC.csv")
df.head(10)


# In[3]:


# create a graph using Source and Target for connections.
# Create the directed and weighted graph 
G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr='p', create_using=nx.DiGraph())


# In[ ]:


# plot network
plt.figure(figsize=(80,70))
nx.draw(G, with_labels=True)
plt.show()


# In[16]:


# Info of network
print(nx.info(G))


# In[5]:


G.is_directed()


# In[6]:


# Check connected or disconnected network --> directed--> Strong/weak connectivity
print(nx.is_strongly_connected(G))
print(nx.is_weakly_connected(G))

#Returns number of strongly connected components in graph.
print(nx.number_strongly_connected_components(G))

#Generate nodes in strongly connected components of graph.
SG = nx.strongly_connected_components(G)

#Returns the number of weakly connected components in G.
print(nx.number_weakly_connected_components(G))


# In[5]:


# Generate connected components and select the largest:
largest_component = max(nx.weakly_connected_components(G), key=len)
len(largest_component) 
Gconnected = G.subgraph(largest_component)
print(nx.info(Gconnected))



# In[10]:


# Calculate degree
print("Node Degree")
for v in Gconnected:
    print(f"{v:4} {Gconnected.degree(v):6}")


# In[11]:


# Calculate degrees and frequencies
degree_sequence = sorted(dict(nx.degree(Gconnected)).values(),reverse=False) # degree sequence

degreeCount = collections.Counter(degree_sequence)
degrees, frequency = zip(*degreeCount.items())

print(degrees)
frequency


# In[14]:


# Plot degree distribution
figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
plt.grid(True)
plt.plot(degrees, frequency, 'ro-')
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree distribution")
plt.savefig('degree')
plt.show()


# In[16]:


figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
plt.grid(True)
plt.xlabel("Log Degree")
plt.ylabel("Frequency")
plt.title("Log-Log scale Degree distribution")
plt.loglog(degrees, frequency,'ro-')
plt.savefig('log')
plt.show()


# In[18]:


k = sum(dict(nx.degree(Gconnected)).values())/float(len(Gconnected))
n = len(Gconnected)
# Calculate the assuming average clustering coefficient the network is random
k/n


# In[19]:


print(nx.average_shortest_path_length(Gconnected))


# In[18]:


# Make a list of the S, we'll use it later
S = list(df.Source.unique())
S


# In[19]:


# Make a list of the clubs, we'll use it later
T = list(df.Target.unique())
T


# In[20]:


# Node
nodes = list(Gconnected.nodes)
len(nodes)

df = pd.DataFrame(nodes)
df.to_csv("./node.csv", sep=',',index=False) 


# In[21]:


# How many connections does BRCA2 have coming out of it?
Gconnected.degree('BRCA2')


# In[22]:


[Gconnected.degree(Target) for Target in T]


# In[23]:


# Clustering coefficient of all nodes
clust_coefficients = nx.clustering(Gconnected)
clust_coefficients


# In[24]:


# Average clustering coefficient
avg_clust = sum(clust_coefficients.values()) / len(clust_coefficients)
print(avg_clust)


# In[26]:


# Calculate the average shortest path assuming the network is random
ln = np.log(n)
lk =  np.log(k)
ln/lk


# In[9]:


#Centrality
#Degree
deg_centrality = nx.degree_centrality(Gconnected)
de_centrality = list(deg_centrality.values())

np.mean(de_centrality)


# In[29]:


df = pd.DataFrame(list(deg_centrality.items()))
df.to_csv("./deg_centrality.csv", sep=',',index=False) 


# In[30]:


indeg = nx.in_degree_centrality(Gconnected)
indeg = list(indeg.values())

print(np.mean(indeg))


# In[10]:


#Closeness
close_centrality = nx.closeness_centrality(Gconnected) 
cl_centrality = list(close_centrality.values())
cl1 = np.mean(cl_centrality)
print(cl1)


# In[32]:


df = pd.DataFrame(list(close_centrality.items()))
df.to_csv("./close_centrality1.csv", sep=',',index=False) 


# In[11]:


#(Shortest Path) Betweenness
bet_centrality = nx.betweenness_centrality(Gconnected, normalized = True, endpoints = False) 
bt_centrality = list(bet_centrality.values())
print(np.mean(bt_centrality ))


# In[12]:


df = pd.DataFrame(list(bet_centrality.items()))
df.to_csv("./bet_centrality.csv", sep=',',index=False) 


# plt.figure(figsize =(15, 10)) 
# plt.plot(de_centrality, 'r', label = 'Degree centrality')
# plt.plot(cl_centrality, 'b' , label = 'Closeness centrality')
# plt.plot(bt_centrality, 'g' , label = 'Betweenness centrality')
# plt.title('Copmarsion Centrality')
# plt.xlabel('Node Number')  
# plt.ylabel('Centrality')
# plt.legend(loc='upper center')
# plt.show()

# In[43]:


r_cl,bt = np.corrcoef(cl_centrality, bt_centrality)
print(r_cl,bt)

r_cl,de = np.corrcoef(cl_centrality, de_centrality)
print(r_cl,de)

r_bt,de = np.corrcoef(bt_centrality, de_centrality)
print(r_bt,de)
scipy.stats.pearsonr(bt_centrality, cl_centrality)[0]    # Pearson's r


# In[44]:


plt.figure(figsize =(15, 10)) 
plt.plot(de_centrality,cl_centrality, 'bo')
plt.title('Degree and Closeness scatterplot')
plt.xlabel('Degree centrality')  
plt.ylabel('Closeness centrality')
plt.savefig('deg-close')
plt.show()


# In[46]:


plt.figure(figsize =(15, 10)) 
plt.plot(de_centrality,bt_centrality, 'bo')
plt.title('Degree and Betweenness scatterplot')
plt.xlabel('Degree centrality')  
plt.ylabel('Betweenness centrality')
plt.savefig('deg-bet')
plt.show()


# In[47]:


plt.figure(figsize =(15, 10)) 
plt.plot(cl_centrality,bt_centrality, 'bo')
plt.title('Closeness and Betweenness scatterplot')  
plt.xlabel('Closeness centrality')
plt.ylabel('Betweenness centrality')
plt.savefig('clos-bet')
plt.show()


# In[48]:


gdfgein = pd.DataFrame(de_centrality)
Dgr = gdfgein[0:2000]
Dgr


# In[13]:


print(max(close_centrality.values()))
print(max(deg_centrality.values()))
print(max(bet_centrality.values()))

print(max(close_centrality.items(), key=operator.itemgetter(1))[0])
print(max(deg_centrality.items(), key=operator.itemgetter(1))[0])
print(max(bet_centrality.items(), key=operator.itemgetter(1))[0])


# In[20]:


dr = pd.read_csv('driver_found.csv')

print('\n confussion matrix Alg1:\n',confusion_matrix(dr['TRUE-Label'], dr['pred1']))

print('\n clasification report for Alg1:\n', classification_report(dr['TRUE-Label'], dr['pred1']))

print('Accuracy1:',accuracy_score(dr['TRUE-Label'], dr['pred1']))
print('Accuracy2:',accuracy_score(dr['TRUE-Label'], dr['pred2']))
print('F1 score1:', f1_score(dr['TRUE-Label'], dr['pred1']))
print('F1 score2:', f1_score(dr['TRUE-Label'], dr['pred2']))
print('Recall1:', recall_score(dr['TRUE-Label'], dr['pred1']))
print('Recall2:', recall_score(dr['TRUE-Label'], dr['pred2']))
print('Precision1:', precision_score(dr['TRUE-Label'], dr['pred1']))
print('Precision2:', precision_score(dr['TRUE-Label'], dr['pred2']))

print('\n confussion matrix Alg2:\n',confusion_matrix(dr['TRUE-Label'], dr['pred2']))

print('\n clasification report for Alg2:\n', classification_report(dr['TRUE-Label'], dr['pred2']))



# In[6]:


g = Gconnected.to_undirected()
print(nx.info(g))
type(g)


# In[7]:


nx.write_gml(g,'graph.gml') # Export NX graph to file


# In[8]:


# load data and graph
g = Graph.Read_GML('graph.gml')
print(g.ecount())
print(g.vcount())
type(g)
g.es["weight"] = g.es["p"]
del g.es["p"]
g.is_weighted()



# In[9]:


# method 1
# community detection with louvain algorithm
louvain_partition = g.community_multilevel()
modularity1 = g.modularity(louvain_partition)
print("The modularity Q based on networkx is {}".format(modularity1))

print(louvain_partition)


# In[28]:


# method 2
# community detection with greedy algorithm
gready = g.community_fastgreedy()

clusters2 = gready.as_clustering()
modularity2 = clusters2.modularity
print(len(clusters2))
modularity2


# In[11]:


# method 3
prop = g.community_label_propagation(initial=None, fixed=None)

modularity3 = g.modularity(prop)
print(prop)


# In[23]:


print(modularity3)


# In[12]:


igraph.plot(clusters2 )


# In[13]:


# modularity rate
print(modularity2)


# In[14]:


# visualize Dendrogram
print(gready)


# In[40]:


##The number of components of each community in louvian Alg
com1 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 0 : 
        com1.append(i)
print(len(com1))

com2 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 1 : 
        com2.append(i)
print(len(com2))

com3 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 2 : 
        com3.append(i)
print(len(com3))

com4 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 3 : 
        com4.append(i)
print(len(com4))

com5 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 4 : 
        com5.append(i)        
print(len(com5))
        
com6 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 5 : 
        com6.append(i)
print(len(com6))
        
com7 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 6 : 
        com7.append(i)
print(len(com7))
        
com8 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 7 : 
        com8.append(i)
print(len(com8))

com9 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 8 : 
        com9.append(i)
print(len(com9))
            
com10 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 9 : 
        com10.append(i)
print(len(com10))

com11 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 10 : 
        com11.append(i)
print(len(com11))

com12 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 11 : 
        com12.append(i)
print(len(com12))

com13 = [] 
for i in range(0, len(louvain_partition.membership)) : 
    if louvain_partition.membership[i] == 12 : 
        com13.append(i)
print(len(com13))


# In[16]:


membersof = clusters2.membership
partitiond = {i: member for i,member in enumerate(membersof)}
type(partitiond)

print(g.clusters())


# In[17]:


clusters2


# In[19]:


# to find indices for 3 
com1 = [] 
for i in range(0, len(clusters2.membership)) : 
    if clusters2.membership[i] == 0 : 
        com1.append(i)     
print(len(com1))


com2 = [] 
for i in range(0, len(clusters2.membership)) : 
    if clusters2.membership[i] == 1 : 
        com2.append(i) 
print(len(com2))


com3 = [] 
for i in range(0, len(clusters2.membership)) : 
    if clusters2.membership[i] == 2 : 
        com3.append(i) 
print(len(com3))


com4 = [] 
for i in range(0, len(clusters2.membership)) : 
    if clusters2.membership[i] == 3 : 
        com4.append(i) 
print(len(com4))


com5 = [] 
for i in range(0, len(clusters2.membership)) : 
    if clusters2.membership[i] == 4 : 
        com5.append(i) 
print(len(com5))


com6 = [] 
for i in range(0, len(clusters2.membership)) : 
    if clusters2.membership[i] == 5 : 
        com6.append(i) 
print(len(com6))


com7 = [] 
for i in range(0, len(clusters2.membership)) : 
    if clusters2.membership[i] == 6 : 
        com7.append(i) 
print(len(com7))


# In[20]:


com1


# In[21]:


print(clusters2.membership)

