#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Data processing
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import operator


# Similarity
from sklearn.metrics.pairwise import cosine_similarity

# Read in data
ratings_df=pd.read_csv('ratings_small.csv')
movies_df = pd.read_csv('movie.csv')

# Merge ratings and movies datasets
df = pd.merge(ratings_df, movies_df, on='movieId', how='inner')

### Item based Collaborative filtering

# Aggregate by movie
aggregate_ratings = df.groupby('title').agg(mean_rating=('rating', 'mean'),
                                            number_of_ratings=('rating', 'count'),
                                            genres=('genres', '|'.join)).reset_index()

#Keep the movies with 100 ratings
agg_ratings_MT100 = aggregate_ratings[aggregate_ratings['number_of_ratings']>100]


# Merge data
df_Filtered = pd.merge(df, agg_ratings_MT100[['title']], on='title', how='inner')

matrix = df_Filtered.pivot_table(index='title', columns='userId', values='rating')

matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)

# Item similarity matrix using Pearson correlation
similarity_score= matrix_norm.T.corr()

cosine_score = cosine_similarity(matrix_norm.fillna(0))

# Item-based recommendation function
# Recommendation function considering genre preferences and user ID
def item_based_rec(picked_user_id, number_of_similar_items, number_of_recommendations):
    # Get unwatched movies for the user
    picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_user_id].isna()).reset_index()
    picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[picked_user_id] == True]['title'].values.tolist()
    picked_userid_watched = pd.DataFrame(matrix_norm[picked_user_id].dropna(axis=0, how='all') \
                                        .sort_values(ascending=False)).reset_index() \
                                        .rename(columns={picked_user_id: 'rating'})

    rating_prediction = {}

    for picked_movie in picked_userid_unwatched:
        # Extract genre information for the picked movie
        picked_movie_genre = movies_df[movies_df['title'] == picked_movie]['genres'].values[0]

        # Filter movies in the same genre as the picked movie
        same_genre_movies = movies_df[movies_df['genres'].str.contains(picked_movie_genre)]

        # Ensure the titles in the user's watched movies match the titles in the similarity_score matrix
        user_watched_same_genre = picked_userid_watched[picked_userid_watched['title'].isin(same_genre_movies['title'])]

        # Calculate the predicted rating for the picked movie in the same genre
        if not user_watched_same_genre.empty:
            weighted_ratings = user_watched_same_genre['title'].apply(
                lambda x: picked_userid_watched[picked_userid_watched['title'] == x]['rating'].values[0] *
                          similarity_score[x][picked_movie]).sum()

            weighted_similarities = user_watched_same_genre['title'].apply(
                lambda x: similarity_score[x][picked_movie]).sum()

            predicted_rating = weighted_ratings / weighted_similarities
        else:
            predicted_rating = 0  # Default rating if there are no similar movies watched

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

def recommend_movies_by_genre(input_genres, movies_df, tfidf_matrix_combined, knn_model, num_recommendations=5):
    # Filter movies by selected genres
    selected_movies = movies_df[movies_df['genres'].str.contains('|'.join(input_genres))]

    # Get movie IDs for selected movies
    movie_ids = selected_movies['movieId'].values

    # Content-based recommendations
    recommended_movies_content = []

    for movie_id_user in movie_ids:
        # Get the index of the chosen movie
        movie_index_content = movies_df[movies_df['movieId'] == movie_id_user].index[0]
        distances_content, indices_content = knn_model.kneighbors(tfidf_matrix_combined[movie_index_content], n_neighbors=num_recommendations + 1)
        recommended_movies_content.extend([(movies_df.iloc[idx]['title'], 1 - distances_content.flatten()[i]) for i, idx in enumerate(indices_content.flatten()[1:])])

    # Sort recommended movies by similarity score and limit to the top 5
    recommended_movies_content.sort(key=lambda x: x[1], reverse=True)
    recommended_movies_content = recommended_movies_content[:num_recommendations]

    return recommended_movies_content

# Extract distinct genres from the dataset
distinct_genres = movies_df['genres'].str.split('|').explode().unique()

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
# Allow users to select one or more genres
selected_genres = st.sidebar.multiselect('Select one or more genres', distinct_genres)

# Content-based recommendations
recommended_movies_content = []

# Get recommendations
if st.button("Get Recommendations"):
    # Get movie recommendations based on genres
    recommended_movies_content = recommend_movies_by_genre(selected_genres, movies_df, tfidf_matrix_combined, knn_model)

    # Collaborative filtering recommendations
    recommended_movie_ratings = item_based_rec(picked_user_id=user_id, number_of_similar_items=5, number_of_recommendations=5)

    # Display recommendations
    st.subheader("Explore Unwatched Gems Based on Your Ratings:", divider='green')
    for movie, rating in recommended_movie_ratings:
        st.write(movie)
        
        
    st.subheader(f"Recommendations Based on Your Choice of Genres: {selected_genres}", divider='blue')
    if recommended_movies_content:
        for title, score in recommended_movies_content:
            st.write(title)
    else:
        st.write("No recommendations found for the selected genres.")
    

# User feedback
    st.divider()
    st.subheader("User Feedback")
    user_feedback = st.selectbox("How would you rate the recommendations we made for you?", [5, 4, 3, 2, 1])

# You can save the feedback in your dataset for future model improvement
    st.write(f"Thank you for your feedback! You rated the recommendations as: {user_feedback}")

