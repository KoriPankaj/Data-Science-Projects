# Assignment Data-Preprocessing 
#### STANDARDIZATION AND NORMALIZATION #####
import pandas as pd # deals dataframe
import numpy as np # deals with numeric imputation

### Standardization
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\Seeds_data.csv") # read the csv file

a = df.describe() # gives the basuc details of distribution

# Initialise the Scaler
scaler = StandardScaler() 
# To scale data
std_x = scaler.fit_transform(df.iloc[:,0:7]) # scale free + transfrom to array
std_y = df['Type'] 
# Convert the array back to a dataframe
dfstd_x = pd.DataFrame(std_x) # convert array to dataframe
df_std = pd.concat([dfstd_x,std_y], axis = 1) # concatenate x and y columnwise
res_std = dataset.describe() # after standardization 


### Normalization
## load data set
df_1 = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\Seeds_data.csv")
df_1.columns
df_1.drop(['Type'], axis = 1, inplace = True) # drop the labelled column

a1 = df_1.describe()

### Normalization function - built a user-defined function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_1norm = norm_func(df_1) # apply the min-max scaler
df_norm = pd.concat([df_1norm,df['Type']], axis =1) # concatenate the columns and obtain final normalized dataset.

res_norm = df_norm.describe() 
