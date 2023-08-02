############# DATA MINING ############
## HIERARCHIAL CLUSTERING ##
# Q-1 EastWest Airlines data
import pandas as pd
import matplotlib.pylab as plt

Air_EW = pd.read_csv(r"E:\360digitMG\ASsignments\6.Data min- Hirer_clust\EastWestAirlines.csv")

Air_EW.describe()
Air_EW.info()


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Air_EW.iloc[:, 1:])
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


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

Air_EW['clust'] = cluster_labels # creating a new column and assigning it to new column 

Air_EW1 = Air_EW.iloc[:, [0,12,1,2,3,4,5,6,7,8,9,10,11]] # Relocating index name
Air_EW1.head()

# Aggregate mean of each cluster
clust_df = Air_EW1.iloc[:, 2:12].groupby(Air_EW1.clust).mean() # Take mean of each cluster 0 to 8

# creating a csv file    
Air_EW1.to_csv(r'E:\360digitMG\ASsignments\6.Data min- Hirer_clust\Airlines.csv', encoding = "utf-8")

import os
os.getcwd()
