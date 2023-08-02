###### ASSIGNMENT DATAPREPROESSING #######

# Type casting #
import pandas as pd

data = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets/OnlineRetail.csv",encoding ='unicode_escape')
data.dtypes
# Q1 convert 'float64' into 'int64' type.###################

data.UnitPrice = data.UnitPrice.astype('int64')
data.dtypes

# Q2 Check for duplicates and handle the duplicate.#############
# Identify duplicates records in the data.

data = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets/OnlineRetail.csv",encoding ='unicode_escape')
duplicate = data.duplicated()
duplicate
sum(duplicate)

# Removing Duplicates.
data1 = data.drop_duplicates()
# Q3 ###############
# EXPLORATORY DATA ANALYSIS
import matplotlib.pyplot as plt 

data.describe()
plt.boxplot(data.Quantity)
# Value of Quantity in -ve also means some return products are also considered.
# 50 % of quantity are distributed between 1 to 10
plt.boxplot(data.UnitPrice)
# Here also min value is in -ve means return transaction are in the data.
# 50 % of Unit price are distributed between 1 to 4.15
plt.hist(data.Quantity,edgecolor="red", bins=10000)
# with the help of histogram we can visualize majority of the data distribution
plt.hist(data.UnitPrice,edgecolor="red", bins=100)
help(plt.hist)

plt.scatter(data.Quantity,data.UnitPrice)
# highest unit prices are for the single quantity.