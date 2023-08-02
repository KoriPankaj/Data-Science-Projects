############# DATA MINING ############
## HIERARCHIAL CLUSTERING ##
# Q-4 Autoinsurance dataset
import pandas as pd
import matplotlib.pylab as plt

Autoins_data = pd.read_csv(r"E:\360digitMG\ASsignments\6.Data min- Hirer_clust\AutoInsurance.csv")

Autoins_data.describe()
Autoins_data.info()  # Details about missing values

# Arrange the dataset, numeric data are on one side and categorical are on another.
Autoins_df = Autoins_data[['Customer','State','Response','Coverage','Education','Effective To Date','EmploymentStatus','Gender','Location Code',
                           'Marital Status','Policy Type','Policy','Renew Offer Type','Sales Channel','Vehicle Class','Vehicle Size',
                           'Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints',
                           'Number of Policies','Total Claim Amount']]
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Autoins_df.iloc[:, 16:]) # scaled numeical data
df_norm.describe()++
x_uni = Autoins_df.iloc[:,0] # Unique Attribute
y_cat = Autoins_df.iloc[:,1:16] # categorical data (string)
Autoins_dfscl = pd.concat([x_uni,y_cat,df_norm], axis =1)


# Data is mixture of both categorical & numerical data.
# we will take only categorical part of dataset and convert to numerical dataset

# Label Encoder Techniques
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
Lbenc = LabelEncoder()
for i in y_cat:
    a = y_cat[i]
    y_cat_nu[i] = Lbenc.fit_transform(a)

# Make a Final dataset for Hierarchial clustering after one hot encoding operation.
autoins_final = pd.concat([x_uni,y_cat_nu,df_norm], axis =1)
autoins_final.columns
autoins_final.var() # Check zero variance feature

# Not taking the string dataset.
autoins_clust = autoins_final.iloc[:,1:]# Cannot take the unique column (string)
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(autoins_clust, method = "complete", metric = "euclidean") # import linkage function , Linkage measure = complete, Distance measure = euclidean 

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show() # helps to select no of clusters


# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(autoins_clust) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
autoins_final.insert(1, 'clust', cluster_labels) # adding a new column(cluster_labels) and assigning it to new column 

autoins_final.head()
# Aggregate mean of each cluster
clust_df = autoins_final.iloc[:, 2:25].groupby(autoins_final.clust).mean() # Take mean of each cluster 0 to 4

# creating a csv file    
autoins_final.to_csv(r'E:\360digitMG\ASsignments\6.Data min- Hirer_clust\autoinsurance_clust.csv', encoding = "utf-8")

import os
os.getcwd()
