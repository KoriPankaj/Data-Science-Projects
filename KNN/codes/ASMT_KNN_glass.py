############# DATA MINING ############
##### SUPERVISED LEARNING ######
#### K-NEAREST NEIGHBOUR ####
# Q-1 GLASS DATASET

import pandas as pd
import numpy as np

glass_df = pd.read_csv(r"E:\360digitMG\ASsignments\KNN\glass.csv")
glass_df.head


desc = glass_df.describe() # descriptive statistical information
glass_df.isnull().sum() # checking null values

df = glass_df.drop('Type',axis =1)
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df)
df_norm.describe()

X = np.array(df_norm) # Predictors 
Y = np.array(glass_df['Type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Imbalance check
glass_df.Type.value_counts()

ytrain = pd.DataFrame(Y_train)
ytest = pd.DataFrame(Y_test)

ytrain.value_counts()
ytest.value_counts()

from sklearn.neighbors import KNeighborsClassifier

acc = []
for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")

# Training and test accuracy 
# at k= 15 training accuracy & test accuracy is closest and high
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train, Y_train) # apply KNN-model to training data

pred = knn.predict(X_test) # prediction on test data
pred   

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred)) # calculating test accuracy
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) # test accuracy


# error on train data
pred_train = knn.predict(X_train) # prediction on training data
print(accuracy_score(Y_train, pred_train)) # calculating training accuracy
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) # training accuracy



