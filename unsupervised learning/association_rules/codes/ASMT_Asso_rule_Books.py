############# DATA MINING ############
###### UNSUPERVSISED LEARNING ######
### ASSOCIATION RULES #####


# Q-1 Books Dataset

# conda install mlxtend
# or
pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
# Creating Data Frame for the transactions data
book_df = pd.read_csv(r"E:\360digitMG\ASsignments\9.Association rules\book.csv")

frequent_itemsets = apriori(book_df, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
colors = ["red","green","blue","black","yellow","magenta","cyan"]
plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color = colors)
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
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

''' "CookBks" & "ChildBks" are the highest selling products both have high support value, and support for their combnation is the highest amongst the others.
from 2 item list & 3 item list "ItalAtlas" and "ItalArt" giving the least confidence of buying amongst other transactions.
from 4 item list it is found that "ItalArt" has highest lift ratio of 20 if someone buys "ArtBks" followed by some others, same scenario observed
for "ItalCook" & "ItalAtlas" if any of them is in the basket then customer will buy "ItalArt"Book.
From rules it can be conclude "ItalAtlas" and "ItalArt" least selling books 
from 2 item list it is found customer was frequent buy "childBks","cookBks", "DoltYBks" and "GeoBks".'''



