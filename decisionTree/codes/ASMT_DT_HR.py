
####### DATA MINING #########
### SUPERVISED LEARNING ###
## DECISION TREE  & RANDOM FOREST ##

# Q4. HR dataset #

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"E:\360digitMG\ASsignments\Decision Tree\HR_DT.csv")

data.isnull().sum() # checking any null values
data.columns
data.describe()  
data1 = data

# Converting into discrete/Numerical
data1['no of Years of Experience of employee'] = data1['no of Years of Experience of employee'].astype(int)
# Monthly income into binary data
data1[' monthly income of employee'] = pd.cut(data1[' monthly income of employee'], bins=[min(data1[' monthly income of employee'])-1,data1[' monthly income of employee'].mean(), max(data1[' monthly income of employee'])],labels=["low","good"])
lb = LabelEncoder()
data1['Position of the employee'] = lb.fit_transform(data1['Position of the employee']) 
data1.head()

#data1["default"]=lb.fit_transform(data1["default"])

data1[' monthly income of employee'].unique()
data1[' monthly income of employee'].value_counts() # checcking dataset balancing [50:50]
colnames = list(data1.columns)

predictors = colnames[:2]
target = colnames[2]

# Splitting data1 into training and testing data1 set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data1, test_size = 0.25) # 25:75

from sklearn.tree import DecisionTreeClassifier as DT # import decision tree classifier

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target]) # fit model to training data


# Prediction on Test data1
preds = model.predict(test[predictors]) 
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test data1 Accuracy 

# Prediction on Train data1
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train data1 Accuracy
# model as training accuracy & test accuracy are close to each other
''' As Test accuracy = 97.9% & training accuracy = 98.6% i.e Decision Tress model is good fit for HR dataset'''

# Using RandomForest Classifier

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42) 

rf_clf.fit(train[predictors], train[target]) # Train Rf classifier on training data

from sklearn.metrics import accuracy_score, confusion_matrix # Import confusion matrix and accuracy score

confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

# Evaluation on Training Data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))
''' As Test accuracy = 97.9% & training accuracy = 98.6% i.e Decision Tress model is good fit for HR dataset'''