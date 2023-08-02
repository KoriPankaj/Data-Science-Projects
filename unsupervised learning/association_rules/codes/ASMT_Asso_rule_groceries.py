############# DATA MINING ############
###### UNSUPERVSISED LEARNING ######
### ASSOCIATION RULES #####


# Q-2 Groceries Dataset
# pip install mlxtend

import pandas as pd # For dataframes
from mlxtend.frequent_patterns import apriori, association_rules # import Associatiion rules from mlextend library

groceries = [] # create empty list
with open(r"E:\360digitMG\ASsignments\9.Association rules\groceries.csv") as f:
    groceries = f.read()

# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = [] # Create Empty list
for i in groceries:
    groceries_list.append(i.split(",")) # separate groceries using "," and make a list

all_groceries_list = [i for item in groceries_list for i in item]  # make a list of each items 'groceries_list'

from collections import Counter # 

item_frequencies = Counter(all_groceries_list) # count frequnecy of items

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1]) #  make a list of items with their frequencies

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt
colors = ["red","green","blue","black","yellow","magenta","cyan"]
plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = colors) # Bar height from frequencies for 11 items
plt.xticks(list(range(0, 11), ), items[0:11]) # groceries name from items for 11 item
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list))  # convert list to dataframe
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction

groceries_series.columns = ["transactions"] # total 9834 transactions

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*') #  169 columns(items) for 9834 transaction

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True) # Consider transaction have support greater than 0.0075

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True) # sorting accrding to support value

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color = colors)
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) # setup rules on measure of lift ratio
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)  # Bring antecedents & consequents in series

ma_X = ma_X.apply(sorted) # sorted alphabatically

rules_sets = list(ma_X)  # Convert to list

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]  # Bringout the unique comibination

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i)) # Indexing of unique_rules_sets from rules_sets

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to lift and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10) # Sorting according to lift ratio

''' From different rules we can perfrom the market basket analysis 
'Whole milk' the highest number of transactions and 'other vegetabkes', 'rolls/buns' , 'yogurt' has higher chance of buying after whole milk
'root vegetables' have least buying groceries but they have high confidence of buying 'beef', 'onions' as a consequent items. 
pip fruits and tropical fruits is one of the combination having significant value of support and lift ratio '''


