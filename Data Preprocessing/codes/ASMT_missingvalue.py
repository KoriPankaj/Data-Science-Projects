import numpy as np
import pandas as pd

# load the dataset
# use modified ethnic dataset
df = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\claimants.csv")

df.isna().sum() # Total number of NA values for each columns

# Create an imputer object that fills 'Nan' values

# for Mean, Meadian, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer
# Mean and Median imputer are used for numeric data (Claimants age)
# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["CLMAGE"] = pd.DataFrame(mean_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isna().sum() # All nan values filled with '28.14'

# Median Imputer
df = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\claimants.csv")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMAGE"].isna().sum()  # all 2 records replaced by median 
df.isna().sum() # All nan values filled with '30'

# Mode Imputer  (Useful for the discrete data) 
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df["CLMSEX"] = pd.DataFrame(mode_imputer.fit_transform(df[["CLMSEX"]])) # Values contain "0" and "1"
df["CLMINSUR"] = pd.DataFrame(mode_imputer.fit_transform(df[["CLMINSUR"]])) # Values contain "0" and "1"
df["SEATBELT"] = pd.DataFrame(mode_imputer.fit_transform(df[["SEATBELT"]])) # Values contain "0" and "1"
df.isnull().sum()  # all claimants sex, insurance, wearing seatbelt.
