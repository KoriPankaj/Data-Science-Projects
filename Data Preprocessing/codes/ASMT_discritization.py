####### DATA PREPROCESSING #########
# ASSIGNMENT DISCRETIZATION 
# Q1 
# BINARIZATION
import pandas as pd
data = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets/iris.csv")
data.head()
data.describe()
# Divide the data(Sepal_Length) into two categories(big/small)
# big(mean to max) & small(min to mean)
data['Sepal_DsLength'] = pd.cut(data['Sepal_Length'], bins=[min(data.Sepal_Length) - 1, 
                                                  data.Sepal_Length.mean(), max(data.Sepal_Length)], labels=["small","big"])
data.head()
data.Sepal_DsLength.value_counts()

# Same binarization will applied to Sepal_Width, Peatal_Length, Petal_Width
data['Sepal_DsWidth'] = pd.cut(data['Sepal_Width'], bins=[min(data.Sepal_Width) - 1, 
                                                  data.Sepal_Width.mean(), max(data.Sepal_Width)], labels=["small","big"])
data.head()
data.Sepal_DsWidth.value_counts()

data['Petal_DsLength'] = pd.cut(data['Petal_Length'], bins=[min(data.Petal_Length) - 1, 
                                                  data.Petal_Length.mean(), max(data.Petal_Length)], labels=["small","big"])
data.head()
data.Petal_DsLength.value_counts()

data['Petal_DsWidth'] = pd.cut(data['Petal_Width'], bins=[min(data.Petal_Width) - 1, 
                                                  data.Petal_Width.mean(), max(data.Petal_Width)], labels=["small","big"])
data.head()
data.Petal_DsWidth.value_counts()

# ROUNDING
# displaying the datatypes
display(data.dtypes)
  
# converting 'Weight' from float to int
data['Sepal_Length'] = data['Sepal_Length'].astype(int)
data['Sepal_Width'] = data['Sepal_Width'].astype(int)
data['Petal_Length'] = data['Petal_Length'].astype(int)
data['Petal_Width'] = data['Petal_Width'].astype(int)
