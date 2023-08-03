import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"E:\360digitMG\ASsignments\Decision Tree\Diabetes.csv")

data.isnull().sum() # checking any null values
data.columns
data.describe() 

data1 = data
data1.head()

data1.iloc[:,8].unique()
data1.iloc[:,8].value_counts() # checcking dataset balancing
colnames = list(data1.columns)

predictors = colnames[:7]
target = colnames[8]

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