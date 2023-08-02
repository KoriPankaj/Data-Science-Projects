### Data Preprocessing ###
# Inferential statistics #
# Q 5 (wieght score dataset)
import pandas as pd
df = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\Assignment_module02.csv")
df.columns

# Mean
df.Points.mean() # gives the mean 
df.Score.mean()
df.Weigh.mean()

# Median
df.Points.median() # gives the median 
df.Score.median()
df.Weigh.median()

# Mode
df.Points.mode()  # give mode value
df.Score.mode()
df.Weigh.mode()
# Alternaitve method
from scipy import stats
stats.mode(df.Points) # give mode value + frequency of mode value.
stats.mode(df.Score)
stats.mode(df.Weigh)

# Measures of Dispersion / Second moment business decision
df.Points.var() # gives the variance
df.Points.std() # gives the standard deviation
 
df.Score.var()
df.Score.std()

df.Weigh.var()
df.Weigh.std()

# Range
Range_P = max(df.Points) - min(df.Points)
Range_S = max(df.Score) - min(df.Score)
Range_W = max(df.Weigh) - min(df.Weigh)



# Q 7

# Create a Dataframe
data_tb = {'Name of company' :['Allied Signal','Bankers Trust','General Mills','ITT Industries','J.P.Morgan & Co.','Lehman Brothers','Marriott','MCI','Merrill Lynch','Microsoft','Morgan Stanley','Sun Microsystems','Travelers','US Airways','Warner-Lambert'], 
        'Measure_X(%)' :[24.23, 25.53, 25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00]}
                            
import pandas as pd
import numpy as np
df  = pd.DataFrame(data_tb)
df.dtypes
# Visualize Data
import matplotlib.pyplot as plt
plt.hist(df['Measure_X(%)']) # right tailed/ right skewed (Highly) /+ve skewed.
plt.boxplot(df['Measure_X(%)']) # Upper whisker is bigger( Right skewed), atleast 1 outlier is present.
df['Measure_X(%)'].skew() # value > 1.0 considered as highly positve skewed.

IQR = df['Measure_X(%)'].quantile(0.75) - df['Measure_X(%)'].quantile(0.25)
lower_limit = df['Measure_X(%)'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Measure_X(%)'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df['Measure_X(%)'] > upper_limit, True, np.where(df['Measure_X(%)'] < lower_limit, True, False)) # let's flag the outliers in the data set (Here 91.36)
 
# Measure of Central Tendency
df['Measure_X(%)'].mean() # average of data
df['Measure_X(%)'].median() # middlemost value
df['Measure_X(%)'].mode() # no repetition

# Measure of Dispersion
df['Measure_X(%)'].var() 
df['Measure_X(%)'].std() # both show high variance of data values from the mean
