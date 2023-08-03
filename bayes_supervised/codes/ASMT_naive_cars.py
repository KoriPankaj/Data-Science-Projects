############# DATA MINING ############
##### SUPERVISED LEARNING ######
#### NAIVE BAYES ####
# Q-2 CARS DATASET

import pandas as pd
import numpy as np

car_data = pd.read_csv(r"E:\360digitMG\ASsignments\Naive bayes\NB_Car_Ad.csv")

car_data.describe()
car_data.info()  # Details about missing values
car_data.drop('User ID', axis =1)
# Separating categorical and numerical values
car_df = car_data[['Gender','Purchased','Age','EstimatedSalary',]]
x_cat = car_df.iloc[:,0:2] # categorical columns
y_num = car_df.iloc[:,2:] # Numerical columns

x_cat_num = pd.get_dummies(x_cat)  # dummy creation for categorical data
df_final = pd.concat([x_cat_num,y_num], axis =1) 

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

scale_df = norm_func(df_final) # Normilization of dataset

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(scale_df, test_size = 0.2)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB # import naive bayes classifier

# Multinomial Naive Bayes
classifier_mb = MB()  
classifier_mb.fit(train_df, train_df.Purchased)  

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_df) 

pd.crosstab(test_pred_m, test_df.Purchased) # only 1 datapoint was wrongly classified

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, test_df.Purchased) 


# Training Data accuracy
train_pred_m = classifier_mb.predict(train_df) 
pd.crosstab(train_pred_m, train_df.Purchased) # 100 % classification accuracy
accuracy_score(train_pred_m, train_df.Purchased) 


''' Classfication accuracy for both training and test data are 100% '''

##################################################################
##################################################################


