# import os
import pandas as pd

# import Dataset 
entm_df = pd.read_csv(r"E:\360digitMG\ASsignments\10.Recommendation engine\Entertainment.csv", encoding = 'utf8')
entm_df.shape # shape
entm_df.columns


from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")   # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
entm_df["Category"].isnull().sum() 
entm_df["Category"] = entm_df["Category"].fillna(" ")
entm_df = entm_df.drop('Id',axis =1)
# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(entm_df.Category) #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 46 unique words in Category columns.
tfidf_matrix

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix) # Cosine measures

# creating a mapping of entm_df name to index number 
entm_df_index = pd.Series(entm_df.index, index = entm_df['Titles']).drop_duplicates()

entm_df_id = entm_df_index["Heat (1995)"]
entm_df_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    entm_df_id = entm_df_index[Name]
    
    # Getting the pair wise similarity score for all the entm_df's with that 
    # entm_df
    cosine_scores = list(enumerate(cosine_sim_matrix[entm_df_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    entm_df_idx  =  [i[0] for i in cosine_scores_N]
    entm_df_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    entm_df_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    entm_df_similar_show["Titles"] = entm_df.loc[entm_df_idx, "Titles"]
    entm_df_similar_show["Score"] = entm_df_scores
    entm_df_similar_show.reset_index(inplace = True)  
    # entm_df_similar_show.drop(["index"], axis=1, inplace=True)
    print (entm_df_similar_show)
    # return (entm_df_similar_show)

    
# Enter your entm_df and number of entm_df's to be recommended 
get_recommendations("Heat (1995)", topN = 10)
entm_df_index["Heat (1995)"]
