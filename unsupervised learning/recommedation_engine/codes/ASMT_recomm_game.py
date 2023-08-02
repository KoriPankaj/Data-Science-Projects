############# DATA MINING ############
###### UNSUPERVSISED LEARNING ######
### RECOMMENDATION ENGINE #####


# Q-1 Anime Dataset
import pandas as pd # for dataframes

# import Dataset 
gam_df = pd.read_csv(r'E:\360digitMG\ASsignments\10.Recommendation engine\game.csv', encoding = 'utf8')
gam_df.shape # shape
gam_df.columns

df = gam.groupby(['game']).mean() # averaging of ratings for different games
df = df.reset_index()
df = df.drop('userId', axis =1)
df_sim = df.drop('game', axis =1)
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(df_sim, df_sim) # Cosine measures

# creating a mapping of anime name to index number 
df_index = pd.Series(df.index, index = df['game']).drop_duplicates()

df_id = df_index["Advance Wars"]
df_id
def get_recommendations(game, topN):    
    # topN = 10
    # Getting the movie index using its title 
    df_id = df_index[game]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id])) # calculate cosine score for userId through game_id(index no.)
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    game_similar_show = pd.DataFrame(columns=["game", "Score"])
    game_similar_show["game"] = df.loc[game_idx, "game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (game_similar_show)
    # return (anime_similar_show)

    
# Enter your game and number of game to be recommended 
get_recommendations("Tony Hawk's Pro Skater 2", topN =50)
df_index["Tony Hawk's Pro Skater 2"]
