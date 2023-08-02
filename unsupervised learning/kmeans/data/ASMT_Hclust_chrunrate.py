############# DATA MINING ############
## HIERARCHIAL CLUSTERING ##
# Q-3 Telecom customer churn dataset
import pandas as pd
import matplotlib.pylab as plt

Churn_data = pd.read_excel(r'E:\360digitMG\ASsignments\6.Data min- Hirer_clust\Telco_customer_churn.xlsx')

Churn_data.describe()
Churn_data.info()  # Details about missing values

# Arrange the dataset, numeric data are on one side and categorical are on another.
Churn_df = Churn_data[['Customer ID','Count','Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges',
                       'Total Long Distance Charges','Total Revenue','Quarter','Referred a Friend','Offer','Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security',
                       'Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Contract','Paperless Billing','Payment Method']]

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Churn_df.iloc[:, 2:12])
df_norm.describe()
x = Churn_df.iloc[:,0:2]
z = Churn_df.iloc[:,12:30]
Churn_dfscl = pd.concat([x,df_norm,z], axis =1)


# Data is mixture of both categorical & numerical data.
# we will take only categorical part of dataset
churn_new = pd.get_dummies(Churn_df.iloc[:,12:30]) # convert categorical to numeric dataset


# Make a Final dataset for Hierarchial clustering after one hot encoding operation.
churn_final = pd.concat([x,df_norm,churn_new], axis =1)
churn_final.var() # Check zero variance feature
churn_final.drop(['Count'], axis=1, inplace=True) # Drop the count column as same value throughout
churn_final.drop(['Quarter_Q3'], axis=1, inplace=True) # Drop the Quarter column as same value throughout



 # Pop out the string dataset.
churn_clust = churn_final.iloc[:,1:]# Cannot take the unique column (string)
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(churn_clust, method = "complete", metric = "euclidean") # import linkage function , Linkage measure = complete, Distance measure = euclidean 

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show() # helps to select no of clusters


# Now applying AgglomerativeClustering choosing 9 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 9, linkage = 'complete', affinity = "euclidean").fit(churn_clust) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
churn_final.insert(1, 'clust', cluster_labels) # adding a new column(cluster_labels) and assigning it to new column 

churn_final.head()
# Aggregate mean of each cluster
clust_df = churn_final.iloc[:, 2:54].groupby(churn_final.clust).mean() # Take mean of each cluster 0 to 8

# creating a csv file    
churn_final.to_csv(r'E:\360digitMG\ASsignments\6.Data min- Hirer_clust\churnrate_clust.csv', encoding = "utf-8")

import os
os.getcwd()
