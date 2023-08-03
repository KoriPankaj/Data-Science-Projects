####### DATA MINING #########
### SUPERVISED LEARNING ###
## DECISION TREE  & RANDOM FOREST ##

# Q1 Cloth company dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"E:\360digitMG\ASsignments\Decision Tree\Company_Data.csv")

data.isnull().sum() # checking any null values
data.columns
data.describe() 
data1 = data.drop(['Education'],axis = 1) # non relevant
data1['Sales'] = data1['Sales'].astype(int)  # convert to discrete
# Converting into discrete/categorical
lb = LabelEncoder()
data1['ShelveLoc'] = lb.fit_transform(data1['ShelveLoc'])
data1['Urban'] = lb.fit_transform(data1['Urban'])
data1.head()

#data1["default"]=lb.fit_transform(data1["default"])

data1['US'].unique()
data1['US'].value_counts() # checcking dataset balancing
colnames = list(data1.columns)

predictors = colnames[:9]
target = colnames[9]

# Splitting data1 into training and testing data1 set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data1, test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test data1
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test data1 Accuracy 

# Prediction on Train data1
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train data1 Accuracy
# Overfit model as training accuracy > test accuracy

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

''' 