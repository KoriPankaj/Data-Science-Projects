############# DATA MINING ############
###### UNSUPERVSISED LEARNING ######
### ASSOCIATION RULES #####

# Q-4 my phone Dataset

# conda install mlxtend
# or
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules # import Associatiion rules from mlextend library
import matplotlib.pyplot as plt
# Creating Data Frame for the transactions data
phone_df = pd.read_csv(r'E:\360digitMG\ASsignments\9.Association rules\myphonedata.csv')
phone_df1 = phone_df.iloc[:,3:]

frequent_itemsets = apriori(phone_df1, min_support = 0.0075, max_len = 4, use_colnames = True)  # Create item frequency sets & Consider transaction have support greater than 0.0075
                                                                                                # using Apriori algorithm
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)  # sorting accrding to support value
colors = ["grey","red","blue","black","yellow","magenta","cyan"]
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

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list) # Bring antecedents & consequents in series


ma_X = ma_X.apply(sorted) # sorted alphabatically

rules_sets = list(ma_X) # Convert to list

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)] # Bringout the unique comibination

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i)) # Indexing of unique_rules_sets from rules_sets

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to lift and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

''' "white" "blue" & "red" are the most demanding phone from the past sales
For 2 item list  = person who buys "blue" will consecuitvely buy either between "white" & "red".
For 3 item list  = person who buys "red" followed by any another product has high confidence to go for "white"  



