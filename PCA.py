#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:28:11 2022

@author: vasanthdhanagopal
"""
############################# PCA with clustering ##############################


import pandas as pd                               # used for data manipualation
import matplotlib.pyplot as plt                   # used for graph plotting
ht = pd.read_csv("copy file path")
ht.info()                                         # informs the dataypes
ht.describe()                                     # tells the mean, max and min values
ht.isnull().sum()                                 # checks the null value and adds it


# The data is checked for gaussian distribution, but most of them are not, also it is not needed for this analysis
# Normal Q-Q plot
# import scipy.stats as stats
# import pylab
# stats.probplot(ht['age'], dist="norm", plot=pylab)
# plt.show()
# import matplotlib.pyplot as plt
# plt.hist(ht['age'])

# Normalization is good to use when you know that the distribution of your data 
# does not follow a Gaussian distribution

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
ht_norm = scaler.fit_transform(ht)
ht_norm = pd.DataFrame(ht_norm)

###################### Perform hierarchical clutering #########################

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch   # for creating dendrogram 
m = linkage(ht_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(m,
    leaf_rotation = 30,  # rotates the x axis labels
    leaf_font_size = 5, # font size for the x axis labels
)
plt.axhline(y=2, color='r', linestyle='--')
plt.show()

# Now applying AgglomerativeClustering - bottom-top approach based on the dedrogram we choose no of clusters as 6
from sklearn.cluster import AgglomerativeClustering

clus = AgglomerativeClustering(n_clusters = 10, linkage = 'complete', affinity = "euclidean").fit(ht_norm) 
clus.labels_

cluster_labels = pd.Series(clus.labels_)

ht['clustH'] = cluster_labels # creating a new column and assigning it to new column 

ht = ht.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

# Aggregate mean of each cluster
ht.iloc[:,:].groupby(ht.clustH).mean()


######################### Perform K-Means clutering ###########################
from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# scree plot or elbow curve 
TWSS = []
k = list(range(1,15))

for i in k:
    kmeans = KMeans(n_clusters = i, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(ht_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'go-.');plt.title("Elbow Plot");plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
plt.show()

# Selecting 6 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 6)
model.fit(ht_norm)

model.labels_ # getting the labels of clusters assigned to each row 
clust_lables = pd.Series(model.labels_)  # converting numpy array into pandas series object 
ht['clust-km'] = clust_lables # creating a  new column and assigning it to new column 
ht = ht.iloc[:, [15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]

######### Now perform PCA and extract three components ########################
from sklearn.decomposition import PCA
pca = PCA(n_components = 14)     
pca_values = pca.fit_transform(ht_norm)

# The amount of variance that each PCA explains is 
var = (pca.explained_variance_ratio_)
varsum = (pca.explained_variance_ratio_.cumsum()) #cumulative variance
varsumper = varsum*100

# PCA weights
pca.components_
pca.components_[0]


# Variance plot for PCA components obtained 
plt.plot(varsum, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5", "comp6", "comp7", "comp8", "comp9", "comp10", "comp11", "comp12", "comp13"
final = pd.concat([ht, pca_data.iloc[:, 0:10]], axis = 1) # take first three components


# Scatter diagram
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))
#final[['comp0', 'comp1', 'comp2']].apply(lambda x: ax.text(*x), axis=1)

#################3 Now apply hierarchical to the first three components


l1 = linkage(final.iloc[:,15:25], method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram-PCA');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(l1,
    leaf_rotation = 30,  # rotates the x axis labels
    leaf_font_size = 5, # font size for the x axis labels
)
plt.axhline(y=2, color='r', linestyle='--')
plt.show()

# Now applying AgglomerativeClustering - bottom-top approach based on the dedrogram we choose no of clusters as 8
from sklearn.cluster import AgglomerativeClustering

clus_hpca = AgglomerativeClustering(n_clusters = 8, linkage = 'complete', affinity = "euclidean").fit(final.iloc[:,15:25]) 
clus_hpca.labels_

cluster_hpca_labels = pd.Series(clus_hpca.labels_)

ht['clustHPCA'] = cluster_hpca_labels # creating a new column and assigning it to new column 

ht = ht.iloc[:, [0,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]

#################3 Now apply K-means to the first three components
# scree plot or elbow curve 
TWSS = []
k = list(range(1,10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final.iloc[:,15:25])
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'bo-.');plt.title("Elbow Plot-PCA");plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 6 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 6)
model.fit(final.iloc[:,15:25])

model.labels_ # getting the labels of clusters assigned to each row 
clp = pd.Series(model.labels_)  # converting numpy array into pandas series object 
ht['clustKMPCA'] = clp # creating a  new column and assigning it to new column 
ht = ht.iloc[:, [0,1,16,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

plt.figure(figsize=(10, 7))  
scatter = plt.scatter(final['comp0'],final['comp1'],model.labels_)
#plt.legend(handles=scatter.legend_elements()[0], labels=model.labels_)
plt.show()


