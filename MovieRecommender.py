# Data processing
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Similarity
from sklearn.metrics.pairwise import cosine_similarity

# Read in data
ratings=pd.read_csv('ratings_small.csv')
movies = pd.read_csv('movie.csv')

# Merge ratings and movies datasets
df = pd.merge(ratings, movies, on='movieId', how='inner')

### Item based Collaborative filtering

# Aggregate by movie
agg_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()

# Keep the movies with over 50 ratings
agg_ratings_F = agg_ratings[agg_ratings['number_of_ratings']>50]

# Merge data
df_Filtered = pd.merge(df, agg_ratings_F[['title']], on='title', how='inner')

matrix = df_Filtered.pivot_table(index='title', columns='userId', values='rating')

matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)

# Item similarity matrix using Pearson correlation
item_similarity = matrix_norm.T.corr()

item_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))

# Item-based recommendation function
def item_based_rec(picked_userid, number_of_similar_items, number_of_recommendations):
  import operator
  # Movies that the target user has not watched
  picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
  picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[picked_userid]==True]['title'].values.tolist()

  # Movies that the target user has watched
  picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')\
                            .sort_values(ascending=False))\
                            .reset_index()\
                            .rename(columns={picked_userid:'rating'})

  # Dictionary to save the unwatched movie and predicted rating pair
  rating_prediction ={}

  # Loop through unwatched movies
  for picked_movie in picked_userid_unwatched:
    # Calculate the similarity score of the picked movie iwth other movies
    picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index().rename(columns={picked_movie:'similarity_score'})
    # Rank the similarities between the picked user watched movie and the picked unwatched movie.
    picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
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
movies['title_and_genre'] =  movies['genres']  + ' ' + movies['title']

# Fit a TfidfVectorizer for content-based filtering based on movie title and genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_combined = tfidf_vectorizer.fit_transform(movies['title_and_genre'].fillna(''))

# Fit a Nearest Neighbors model for content-based filtering
knn_model_content = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
knn_model_content.fit(tfidf_matrix_combined)


# Streamlit App
st.set_page_config(layout="centered")

st.header("Personalized Movie Recommendation Engine",divider="rainbow")

# User input: userId and movie title
user_id = st.number_input("Enter your userId", min_value=1, max_value=670, value=2)
# User input for movie title with autocomplete options
movie_title = st.selectbox("Select a movie title", movies['title'].tolist())

# Get the movieId for the entered movie title
movie_id_user = movies.loc[movies['title'] == movie_title, 'movieId'].values

# Get recommendations

# Content based recommendations
movie_id_user = movie_id_user[0]

# Get the top 10 recommended movies based on content similarity
movie_index_content = movies[movies['movieId'] == movie_id_user].index[0]
distances_content, indices_content = knn_model_content.kneighbors(tfidf_matrix_combined[movie_index_content], n_neighbors=11)
recommended_movies_content = [(movies.iloc[idx]['title'], 1 - distances_content.flatten()[i]) for i, idx in enumerate(indices_content.flatten()[1:])]

# Collaborative filtering recommendations
recommended_movie_ratings = item_based_rec(picked_userid=user_id, number_of_similar_items=5, number_of_recommendations =10)

st.divider()
# Display the filtered recommendations

# Create two columns
col1, col2 = st.columns(2)

# Display the first list in the first column
with col1:
    st.subheader("More like the movie", divider='blue')
    for movie, sim in recommended_movies_content:
        st.write(movie)

# Display the second list in the second column
with col2:
    st.subheader("From your ratings", divider='green')
    for movie, rating in recommended_movie_ratings:
        st.write(movie)

st.divider()
# Allow users to provide feedback
user_feedback = st.selectbox("How would you rate the recommendations?", [5, 4, 3, 2, 1])

# You can save the feedback in your dataset for future model improvement
st.write(f"Thank you for your feedback! You rated the recommendations as: {user_feedback}")



