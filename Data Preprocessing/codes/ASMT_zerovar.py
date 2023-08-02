#### zero variance and near zero variance ######

import pandas as pd
df = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\Z_dataset.csv")
df_1 = df.drop("Id", axis =1)  # drop the unique Id 
df_2 = df_1.drop("colour",axis =1) # drop the string attribute
# scaled the data
from sklearn.preprocessing import normalize # import normalize function from sklearn library
nrmdf = normalize(df_2) # output is in array from
df_scl = pd.DataFrame(nrmdf) # convert to dataframe
df_scl.var() # variance of numeric value

# column "0" (sq.length) and "3"(rec.breadth) has near zero variance i.e 0.0019 & 0.006 respectively
# we can take attribute having variance greater than 0.01 ("1", & "2")
df_0var = df_scl.drop(0, axis =1) # drop the sq.length attribute
df_0varfnl = df_0var.drop(3, axis =1) # drop the rec.breadth attribute

# Prepare the final dataset
df_final = pd.concat([df['Id'], df_0varfnl, df['colour']], axis =1) 
df_final_1 = df_final.rename(columns={1: 'square.breath', 2:'rec.length'})
df_final_1.var()
