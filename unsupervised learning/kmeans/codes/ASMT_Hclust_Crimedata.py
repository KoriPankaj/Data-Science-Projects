############# DATA MINING ############
## HIERARCHIAL CLUSTERING ##
# Q-2 Crime rate data
import pandas as pd
import matplotlib.pylab as plt

CR = pd.read_csv(r"E:\360digitMG\ASsignments\6.Data min- Hirer_clust\crime_data.csv")

CR.describe()
CR.info()  # Details abiut missing values

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(CR.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean") # import linkage function , Linkage measure = complete, Distance measure = euclidean 

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show() # helps to select no of clusters


# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

CR['clust'] = cluster_labels # creating a new column and assigning it to new column 

CR1 = CR.iloc[:, [0,5,1,2,3,4,]] # Relocating index name
CR1.head()
# Aggregate mean of each cluster
clust_df = CR1.iloc[:, 2:6].groupby(CR1.clust).mean() # Take mean of each cluster 0 to 8

# creating a csv file    
CR.to_csv(r'E:\360digitMG\ASsignments\7.Data min- Kmeans\K-meanscrimerate_clust.csv', encoding = "utf-8")

import os
os.getcwd()
