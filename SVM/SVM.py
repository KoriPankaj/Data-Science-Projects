import pandas as pd # 
import numpy as np

letters = pd.read_csv(r"E:\360digitMG\SVM\letterdata.csv")
letters.describe() # dataset discription

from sklearn.svm import SVC # import support Vector classifier
from sklearn.model_selection import train_test_split 

train,test = train_test_split(letters, test_size = 0.20) # 20 % random goes to test and rest to train

train_X = train.iloc[:, 1:] # predictors (input)
train_y = train.iloc[:, 0] # target
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear") # linear kernel
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X) #prediction on test data

np.mean(pred_test_linear == test_y) # compare predicted with actual value and take mean of them

# kernel = rbf
model_rbf = SVC(kernel = "rbf") # kernal radial basis function
model_rbf.fit(train_X, train_y) # build miodel
pred_test_rbf = model_rbf.predict(test_X) 

np.mean(pred_test_rbf==test_y) # compare predicted with actual and take mean of them

pred_train_rbf = model_rbf.predict(train_X) # prediction on training data
np.mean(pred_train_rbf==train_y) # compare predicted train data with actual data and take mean if them
