############# DATA MINING ############
##### SUPERVISED LEARNING ######
#### K-NEAREST NEIGHBOUR ####
# Q-2 ZOO DATASET

import pandas as pd
import numpy as np

zoo_df = pd.read_csv(r"E:\360digitMG\ASsignments\KNN\Zoo.csv")
zoo_df.head


desc = zoo_df.describe() # descriptive statistical information
zoo_df.isnull().sum() # checking null values

df = zoo_df.drop('type',axis =1) # Target column
df = df.drop('animal name',axis =1) # drop unique column
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df)
df_norm.describe()

X = np.array(df_norm) # Predictors 
Y = np.array(zoo_df['type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Imbalance check
zoo_df.type.value_counts()

ytrain = pd.DataFrame(Y_train)
ytest = pd.DataFrame(Y_test)

ytrain.value_counts()
ytest.value_counts()

from sklearn.neighbors import KNeighborsClassifier # Import KNN classifier

acc = []
for i in range(8, 50, 3):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(8,50,3),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(8,50,3),[i[1] for i in acc],"bo-")

# Training and test accuracy 
# at k= 15 training accuracy & test accuracy is closest and high
knn = KNeighborsClassifier(n_neighbors = 8)
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


''' Training accuracy is 91.25% and test accuracy is 85.17% i.e model is showing overing for zoo dataset '''
