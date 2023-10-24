#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data processing
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Similarity
from sklearn.metrics.pairwise import cosine_similarity

# Read in data
ratings_df=pd.read_csv('ratings_small.csv')
movies_df = pd.read_csv('movie.csv')

# Merge ratings and movies datasets
df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')

### Item based Collaborative filtering

# Aggregate by movie
aggregate_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()

# Keep the movies with over 50 ratings
agg_ratings_Filtered = aggregate_ratings[aggregate_ratings['number_of_ratings']>50]

# Merge data
df_Filtered = pd.merge(df, agg_ratings_Filtered[['title']], on='title', how='inner')

matrix = df_Filtered.pivot_table(index='title', columns='userId', values='rating')

matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)

# Item similarity matrix using Pearson correlation
item_similarity = matrix_norm.T.corr()

item_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))

# Item-based recommendation function
def item_based_rec(userid, number_of_similar_items, number_of_recommendations):
  import operator
  # Movies that the target user has not watched
  picked_unwatched = pd.DataFrame(matrix_norm[userid].isna()).reset_index()
  picked_unwatched = picked_unwatched[picked_unwatched[userid]==True]['title'].values.tolist()

  # Movies that the target user has watched
  picked_watched = pd.DataFrame(matrix_norm[userid].dropna(axis=0, how='all')\
                            .sort_values(ascending=False))\
                            .reset_index()\
                            .rename(columns={userid:'rating'})

  # Dictionary to save the unwatched movie and predicted rating pair
  rating_prediction ={}

  # Loop through unwatched movies
  for picked_movie in picked_unwatched:
    # Calculate the similarity score of the picked movie iwth other movies
    picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index().rename(columns={picked_movie:'similarity_score'})
    # Rank the similarities between the picked user watched movie and the picked unwatched movie.
    picked_userid_watched_similarity = pd.merge(left=picked_watched,
                                                right=picked_movie_similarity_score,
                                                on='title',
                                                how='inner')\
                                        .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
    # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
    predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                        weights=picked_userid_watched_similarity['similarity_score']), 3)
    # Save the predicted rating in the dictionary
    rating_prediction[picked_movie] = predicted_rating
    # Return the top recommended movies
  return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]

### Content-based filtering

# Combine title and genre for content-based filtering
movies_df['title_and_genre'] =  movies_df['title']  + ' ' + movies_df['genres']

# Fit a TfidfVectorizer for content-based filtering based on movie title and genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_combined = tfidf_vectorizer.fit_transform(movies_df['title_and_genre'].fillna(''))

# Fit a Nearest Neighbors model for content-based filtering
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model.fit(tfidf_matrix_combined)


# Streamlit App
st.set_page_config(page_title="Movie Recommendation App", page_icon="🎬")
# Title with the movie icon
st.title("🎬 Cinematique: Your Movie Adventure Awaits 🎬")
st.divider()
#st.header("Movie Recommendation App",divider="violet")
st.markdown("Discover personalized movie recommendations!")
# User input
st.sidebar.title("User Inputs")
user_id = st.sidebar.number_input("Enter your User ID", min_value=1, max_value=670, value=2)
movie_title = st.sidebar.selectbox("Select a movie:", movies_df['title'].tolist())

# Get the movieId for the entered movie title
movie_id_user = movies_df.loc[movies_df['title'] == movie_title, 'movieId'].values[0]

# Get recommendations
if st.button("Get Recommendations"):
    # Content-based recommendations
    movie_index_content = movies_df[movies_df['movieId'] == movie_id_user].index[0]
    distances_content, indices_content = knn_model.kneighbors(tfidf_matrix_combined[movie_index_content], n_neighbors=6)
    recommended_movies_content = [(movies_df.iloc[idx]['title'], 1 - distances_content.flatten()[i]) for i, idx in enumerate(indices_content.flatten()[1:])]

    # Collaborative filtering recommendations
    recommended_movie_ratings = item_based_rec(userid=user_id, number_of_similar_items=5, number_of_recommendations=5)

    # Display recommendations
    st.subheader("Explore Unwatched Gems Based on Your Ratings:", divider='green')
    for movie, rating in recommended_movie_ratings:
        st.write(movie)
        
        
    st.subheader(f"Recommendations Based on Your Choice of : {movie_title}", divider='blue')
    for movie, sim in recommended_movies_content:
        st.write(movie)

    

# User feedback
    st.divider()
    st.subheader("User Feedback")
    user_feedback = st.selectbox("How would you rate the recommendations we made for you?", [5, 4, 3, 2, 1])

# You can save the feedback in your dataset for future model improvement
    st.write(f"Thank you for your feedback! You rated the recommendations as: {user_feedback}")

