# Assignment Data-Preprocessing 
#### Data_preparation_outlier_treatment #####
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets/boston_data.csv")


# let's find outliers in Salaries
df.dtypes
df.boxplot()
sns.boxplot(data=df)

# Detection of outliers 
sns.boxplot(df.crim) # Visulaize box plot of "crim" attribute
IQR = df['crim'].quantile(0.75) - df['crim'].quantile(0.25) # Calculate Interquartile range
lower_limit = df['crim'].quantile(0.25) - (IQR * 1.5)  # Lower fence value
upper_limit = df['crim'].quantile(0.75) + (IQR * 1.5)  # Upper fance value
outliers_df = np.where(df['crim'] > upper_limit, True, np.where(df['crim'] < lower_limit, True, False)) # let's flag the outliers in the data set
df_trimmed = df.loc[~(outliers_df), ] # Trimmed dataframe( remove the outliers)
sns.boxplot(df_trimmed.crim)  

sns.boxplot(df.zn)
IQRzn = df['zn'].quantile(0.75) - df['zn'].quantile(0.25)
lower_limitzn = df['zn'].quantile(0.25) - (IQRzn * 1.5)
upper_limitzn = df['zn'].quantile(0.75) + (IQRzn * 1.5)
outliers_zn = np.where(df['zn'] > upper_limitzn, True, np.where(df['zn'] < lower_limitzn, True, False))
df_trimmedzn = df.loc[~(outliers_zn), ]
sns.boxplot(df_trimmedzn.zn)

sns.boxplot(df.indus) # no outliers present no need to perform
IQRind = df['indus'].quantile(0.75) - df['indus'].quantile(0.25)
lower_limitind = df['indus'].quantile(0.25) - (IQRind * 1.5)
upper_limitind = df['indus'].quantile(0.75) + (IQRind * 1.5)
outliers_ind = np.where(df['indus'] > upper_limitind, True, np.where(df['indus'] < lower_limitind, True, False))
df_trimmedind = df.loc[~(outliers_ind), ]
sns.boxplot(df_trimmedind.indus)

sns.boxplot(df.chas) # All values are "0"
IQRchs = df['chas'].quantile(0.75) - df['chas'].quantile(0.25)
lower_limitchs = df['chas'].quantile(0.25) - (IQRchs * 1.5)
upper_limitchs = df['chas'].quantile(0.75) + (IQRchs * 1.5)
outliers_chs = np.where(df['chas'] > upper_limitchs, True, np.where(df['chas'] < lower_limitchs, True, False))
df_trimmedchs = df.loc[~(outliers_chs), ]
sns.boxplot(df_trimmedind.chs)

sns.boxplot(df.chas) # All values are "0"
sns.boxplot(df.nox)  # NO outlier are present
sns.boxplot(df.rm)
IQRrm = df['rm'].quantile(0.75) - df['rm'].quantile(0.25)
lower_limitrm = df['rm'].quantile(0.25) - (IQRrm * 1.5)
upper_limitrm = df['rm'].quantile(0.75) + (IQRrm * 1.5)
outliers_rm = np.where(df['rm'] > upper_limitrm, True, np.where(df['rm'] < lower_limitrm, True, False))
df_trimmedrm = df.loc[~(outliers_rm), ]
sns.boxplot(df_trimmedrm.rm) # df changed so no sense of visualization

sns.boxplot(df.age) # NO outlier are present

sns.boxplot(df.dis)
IQRdis = df['dis'].quantile(0.75) - df['dis'].quantile(0.25)
lower_limitdis = df['dis'].quantile(0.25) - (IQRdis * 1.5)
upper_limitdis = df['dis'].quantile(0.75) + (IQRdis * 1.5)
outliers_dis = np.where(df['dis'] > upper_limitdis, True, np.where(df['dis'] < lower_limitdis, True, False))
df_trimmeddis = df.loc[~(outliers_dis), ]
sns.boxplot(df_trimmeddis.dis)

sns.boxplot(df.rad)# NO outlier are present

sns.boxplot(df.tax)# NO outlier are present

sns.boxplot(df.ptratio)
IQRpr = df['ptratio'].quantile(0.75) - df['ptratio'].quantile(0.25)
lower_limitpr = df['ptratio'].quantile(0.25) - (IQRpr * 1.5)
upper_limitpr = df['ptratio'].quantile(0.75) + (IQRpr * 1.5)
outliers_pr = np.where(df['ptratio'] > upper_limitpr, True, np.where(df['ptratio'] < lower_limitpr, True, False))
df_trimmedpr = df.loc[~(outliers_pr), ]
sns.boxplot(df_trimmedpr.ptratio)

sns.boxplot(df.black)
IQRblk = df['black'].quantile(0.75) - df['black'].quantile(0.25)
lower_limitblk = df['black'].quantile(0.25) - (IQRblk * 1.5)
upper_limitblk = df['black'].quantile(0.75) + (IQRblk * 1.5)
outliers_blk = np.where(df['black'] > upper_limitblk, True, np.where(df['black'] < lower_limitblk, True, False))
df_trimmedblk = df.loc[~(outliers_blk), ]
sns.boxplot(df_trimmedblk.black)

sns.boxplot(df.lstat)
IQRlst = df['lstat'].quantile(0.75) - df['lstat'].quantile(0.25)
lower_limitlst = df['lstat'].quantile(0.25) - (IQRlst * 1.5)
upper_limitlst = df['lstat'].quantile(0.75) + (IQRlst * 1.5)
outliers_lst = np.where(df['lstat'] > upper_limitlst, True, np.where(df['lstat'] < lower_limitlst, True, False))
df_trimmedlst = df.loc[~(outliers_lst), ]
sns.boxplot(df_trimmedlst.lstat)

sns.boxplot(df.medv)
IQRmedv = df['medv'].quantile(0.75) - df['medv'].quantile(0.25)
lower_limitmedv = df['medv'].quantile(0.25) - (IQRmedv * 1.5)
upper_limitmedv = df['medv'].quantile(0.75) + (IQRmedv * 1.5)
outliers_medv = np.where(df['medv'] > upper_limitmedv, True, np.where(df['medv'] < lower_limitmedv, True, False))
df_trimmedmedv = df.loc[~(outliers_medv), ]
sns.boxplot(df_trimmedmedv.medv)


# Antoher method using loop to perform the outlier treatment.
perc = {}
lnj_col = {}

for i, j in df.items():
        q1 = j.quantile(0.25)
        q3 = j.quantile(0.75)
        irq = q3 - q1
        j_col = j[(j <= q1 - 1.5 * irq) | (j >= q3 + 1.5 * irq)]
        lnj_col[i] = len(j_col)  # no. of outliers present in the data for each column
        perc[i] = np.shape(j_col)[0] * 100.0 / np.shape(df)[0] # perct[i] = len(j_col)/len(df) *100
        outliers = np.where(df[i] > q3+1.5*irq, True, np.where(df[i] < q1-1.5*irq, True, False))
        df_trimmed = df.loc[~(outliers), ]
        print(i)
        print("Trimmed feature: ",i,"\nshape of trimmed df: ",df_trimmed.shape) # shape of df after columnwise removing outliers
    
        
        



#  3. Winsorization (Brings the outlier values to the fence values)
# install the package
# pip install feature-engine
from feature_engine.outliers import Winsorizer
# For attribute "crim"
winsor_cr = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['crim'])

df_cr = winsor_cr.fit_transform(df[['crim']])
sns.boxplot(df_cr.crim) # boxlplot after winsorization of "crim"
winsor_cr.left_tail_caps_, winsor_cr.right_tail_caps_#inspect the minimum caps and maximum caps 
# For attribute "zn"
winsor_zn = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['zn'])

df_zn = winsor_zn.fit_transform(df[['zn']])
sns.boxplot(df_zn.zn)
winsor_zn.left_tail_caps_, winsor_zn.right_tail_caps_
# For attribute "indus"
winsor_ind = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['indus'])

df_ind = winsor_ind.fit_transform(df[['indus']])
sns.boxplot(df_ind.indus)
winsor_ind.left_tail_caps_, winsor_ind.right_tail_caps_
# For attribute "chas"
winsor_chas = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['chas'])

df_chas = winsor_chas.fit_transform(df[['chas']])
sns.boxplot(df_chas.chas)
winsor_chas.left_tail_caps_, winsor_chas.right_tail_caps_
# For attribute "nox"
winsor_nox = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['nox'])

df_nox = winsor_nox.fit_transform(df[['nox']])
sns.boxplot(df_nox.nox)
winsor_nox.left_tail_caps_, winsor_nox.right_tail_caps_
# For attribute "rm"
winsor_rm = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['rm'])

df_rm = winsor_rm.fit_transform(df[['rm']])
sns.boxplot(df_rm.rm)
winsor_rm.left_tail_caps_, winsor_rm.right_tail_caps_
# For attribute "age"
winsor_age = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['age'])

df_age = winsor_age.fit_transform(df[['age']])
sns.boxplot(df_age.age)
winsor_age.left_tail_caps_, winsor_age.right_tail_caps_
# For attribute "dis"
winsor_dis = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['dis'])

df_dis = winsor_dis.fit_transform(df[['dis']])
sns.boxplot(df_dis.dis)
winsor_dis.left_tail_caps_, winsor_dis.right_tail_caps_
# For attribute "rad"
winsor_rad = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['rad'])

df_rad = winsor_rad.fit_transform(df[['rad']])
sns.boxplot(df_rad.rad)
winsor_rad.left_tail_caps_, winsor_rad.right_tail_caps_
# For attribute "tax"
winsor_tax = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['tax'])

df_tax = winsor_tax.fit_transform(df[['tax']])
sns.boxplot(df_tax.tax)
winsor_tax.left_tail_caps_, winsor_tax.right_tail_caps_
# For attribute "ptratio"
winsor_ptr = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['ptratio'])

df_ptr = winsor_ptr.fit_transform(df[['ptratio']])
sns.boxplot(df_ptr.ptratio)
winsor_ptr.left_tail_caps_, winsor_ptr.right_tail_caps_
# For attribute "black"
winsor_blk = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['black'])

df_blk = winsor_blk.fit_transform(df[['black']])
sns.boxplot(df_blk.black)
winsor_blk.left_tail_caps_, winsor_blk.right_tail_caps_
# For attribute "lstat"
winsor_lst = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['lstat'])

df_lst = winsor_lst.fit_transform(df[['lstat']])
sns.boxplot(df_lst.lstat)
winsor_lst.left_tail_caps_, winsor_lst.right_tail_caps_
# For attribute "medv"
winsor_mdv = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['medv'])

df_mdv = winsor_mdv.fit_transform(df[['medv']])
sns.boxplot(df_mdv.medv)
winsor_mdv.left_tail_caps_, winsor_mdv.right_tail_caps_