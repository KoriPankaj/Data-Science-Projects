



'''
DATA DESCRIPTION 

Undergrad : person is under graduated or not

Marital.Status : marital status of a person

Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government

Work Experience : Work experience of an individual person

Urban : Whether that person belongs to urban area or not'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"E:\360digitMG\ASsignments\Decision Tree\Fraud_check.csv")

data.isnull().sum() # checking any null values
data.columns
data.describe()  
data1 = data

# Converting into discrete/Numerical
# Taxable income into binary data
data1['Taxable.Income'] = pd.cut(data1['Taxable.Income'], bins=[min(data1['Taxable.Income']),30000, max(data1['Taxable.Income'])],labels=["risky","good"])
lb = LabelEncoder()
data1['Undergrad'] = lb.fit_transform(data1['Undergrad']) 
data1['Marital.Status'] = lb.fit_transform(data1['Marital.Status'])
data1['Taxable.Income'] = lb.fit_transform(data1['Taxable.Income'])

data1.head()

#data1["default"]=lb.fit_transform(data1["default"])

data1['Urban'].unique()
data1['Urban'].value_counts() # checcking dataset balancing [50:50]
colnames = list(data1.columns)

predictors = colnames[:5]
target = colnames[5]

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
# Overfit model as training accuracy > test accuracy


########### DECISION TREE PRUNING ###############
# Train the Regression DT

from sklearn import tree
data2= data1
data2['Urban'] = lb.fit_transform(data2['Urban']) 

regtree = tree.DecisionTreeRegressor(max_depth = 3)
train1, test1 = train_test_split(data2, test_size = 0.25)
regtree.fit(train1[predictors], train1[target])
# Prediction
test_pred = regtree.predict(test1[predictors])
train_pred = regtree.predict(train1[predictors])

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(test1[target], test_pred)
r2_score(test1[target], test_pred)

# Error on train dataset
mean_squared_error(train1[target], train_pred)
r2_score(train1[target], train_pred)

# Rsq of training data closed to Rsq of test data hence the model is exceptable

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42) 

rf_clf.fit(train[predictors], train[target]) # Train Rf classifier on training data

from sklearn.metrics import accuracy_score, confusion_matrix # Import confusion matrix and accuracy score

confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

# Evaluation on Training Data
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))
