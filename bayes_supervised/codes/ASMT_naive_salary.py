############# DATA MINING ############
##### SUPERVISED LEARNING ######
#### NAIVE BAYES ####
# Q-1 SALARY DATASET

import pandas as pd
import numpy as np

salary_train = pd.read_csv(r"E:\360digitMG\ASsignments\Naive bayes\SalaryData_Train.csv")

salary_train.describe()
salary_train.info()  # Details about missing values
# Separating categorical and numerical values
train_df = salary_train[['workclass',	'education','maritalstatus',	'occupation','relationship','race','sex','native','Salary', 'age','educationno',	
                         'capitalgain',	'capitalloss',	'hoursperweek',]]
x_cat = train_df.iloc[:,0:9] # categorical columns
y_num = train_df.iloc[:,9:] # Numerical columns

x_cat_num = pd.get_dummies(x_cat)  # dummy creation for categorical data
train_final = pd.concat([x_cat_num,y_num], axis =1) 

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

scale_train_df = norm_func(train_final) # Normilization of dataset
# Similar steps applied for the test dataset
salary_test = pd.read_csv(r"E:\360digitMG\ASsignments\Naive bayes\SalaryData_Test.csv")

test_df = salary_test[['workclass',	'education','maritalstatus','occupation','relationship','race','sex','native','Salary', 'age','educationno',	
                         'capitalgain',	'capitalloss',	'hoursperweek',]]
t_cat = test_df.iloc[:,0:9]
t_num = test_df.iloc[:,9:]
t_cat_num = pd.get_dummies(t_cat)
test_final = pd.concat([t_cat_num,t_num], axis =1)
scale_test_df = norm_func(test_final)


# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB # import naive bayes classifier

# Multinomial Naive Bayes
classifier_mb = MB()  
classifier_mb.fit(scale_train_df, train_df.Salary)  

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(scale_test_df) 

pd.crosstab(test_pred_m, test_df.Salary) # only 1 datapoint was wrongly classified

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, test_df.Salary) 


# Training Data accuracy
train_pred_m = classifier_mb.predict(scale_train_df) 
pd.crosstab(train_pred_m, train_df.Salary) # 100 % classification accuracy
accuracy_score(train_pred_m, train_df.Salary) 


# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

#formula: 
# P(w|spam) = (num of spam with w + alpha)/(Total num of spam emails + K(alpha))
# K = total num of words in the email to be classified

classifier_mb_lap = MB(alpha = 0.75)
classifier_mb_lap.fit(scale_train_df, train_df.Salary) 

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(scale_test_df)
pd.crosstab(test_pred_lap, test_df.Salary) # accuracy increase due to laplacian smooothing
accuracy_score(test_pred_m, test_df.Salary) 



##################################################################
##################################################################


