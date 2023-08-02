
# Assignment Data-Preprocessing #
#### TRANSFORMATION #####

# Read data into Python
import pandas as pd # to deal with dataframe
calori = pd.read_csv(r"E:\360digitMG\ASsignments\4.Data preprocessing\DataSets\calories_consumed.csv")

import scipy.stats as stats  # import scientific python used to solve scientific & engineering problems
import pylab  # import matplotlib.pyplot and numpy at the same space.

# Checking Whether data is normally distributed
x = calori['Weight gained (grams)']
pylab.figure(1)
stats.probplot(x, dist="norm", plot=pylab) # follow non-linearity


import numpy as np
# Transformation to make attribute normal
pylab.figure(2) # display figure 2 so can compare with figure 1
stats.probplot(np.log(x), dist="norm", plot=pylab) # achieve some linearity

y = calori['Calories Consumed']
pylab.figure(3)
stats.probplot(y, dist="norm", plot=pylab) # follow non-linearity
# Transformation to make attribute normal
pylab.figure(4)
stats.probplot(np.log(y), dist="norm", plot=pylab)
