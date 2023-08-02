import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Using Animal category dataset
df = pd.read_csv(r'E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\animal_category.csv')

df.columns # column names
df.shape # gives shape of column

# drop column
df.drop(['Index'], axis=1, inplace=True) # drop the index column
df.dtypes

# Create dummy variables
df_new = pd.get_dummies(df) # categorical data into numeric data using dummyvariables
df_new_1 = pd.get_dummies(df, drop_first = True)  # drop 1st category of each column (cat,female,No,TypesA)

# we have created dummies for all categorical columns
### One Hot Encoding Technique ###

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
OHenc = OneHotEncoder() # initializing method

Ohenc_df = pd.DataFrame(OHenc.fit_transform(df).toarray()) # fit_transform transform series to array, so pd.DataFrame is used.

######### Label Encoder Techniques ################
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
Lbenc = LabelEncoder()

# Data Split into Input and Output variables
x = df.iloc[:, 0:3] # x = input

y = df['Types'] # y = output
df.columns

x['Animals']= labelencoder.fit_transform(x['Animals']) # labelling column animals in alphabetical order
x['Gender'] = labelencoder.fit_transform(x['Gender']) # same for Gender
x['Homly'] = labelencoder.fit_transform(x['Homly']) # same for Homly

### label encode y ###
y = labelencoder.fit_transform(y) # same for Types
y = pd.DataFrame(y)

### we have to convert y to data frame so that we can use concatenate function
# concatenate X and y
Lbenc_df = pd.concat([x, y], axis =1)

## rename column name
Lbenc_df.columns
Lbenc_df = Lbenc_df.rename(columns={0:'Types'}) # change column name from '0' to 'Types'
