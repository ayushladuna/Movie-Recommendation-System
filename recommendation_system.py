import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

print("Script is running...")

# Step 1: Load and Clean Datasets
movies = pd.read_csv('tmdb_5000_movies.csv', dtype={'id': str}, low_memory=False)
credits = pd.read_csv('tmdb_5000_credits.csv', dtype={'movie_id': str}, low_memory=False)

# Rename 'movie_id' to 'id' in credits
credits.rename(columns={'movie_id': 'id'}, inplace=True)

# Merge datasets on 'id' column
movies = movies.merge(credits, on='id')

# Step 2: Preprocess Relevant Features
def convert(obj):
    """Convert JSON-like strings into a space-separated string of values."""
    try:
        return ' '.join([i['name'] for i in ast.literal_eval(obj)])
    except:
        return ''

def get_director(crew):
    """Extract the director's name from the crew list."""
    try:
        for member in ast.literal_eval(crew):
            if member['job'] == 'Director':
                return member['name']
        return ''
    except:
        return ''

# Extract relevant features
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: ' '.join([i['name'] for i in ast.literal_eval(x)[:3]] if pd.notnull(x) else ''))  # Top 3 cast members
movies['director'] = movies['crew'].apply(lambda x: get_director(x))

# Combine features into a single column
movies['combined_features'] = movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['cast'] + ' ' + movies['director']

# Use 'title_x' to reference the movie title
movies = movies.rename(columns={'title_x': 'title'})

# Drop rows with missing combined features
movies = movies[['title', 'combined_features']].dropna()

# Step 3: Create a Count Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies['combined_features'])

# Step 4: Compute Cosine Similarity
cosine_sim = cosine_similarity(count_matrix)

# Step 5: Recommendation Function
def recommend(movie_title, num_recommendations=25):
    try:
        # Get the index of the movie that matches the title
        movie_idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        # Get similarity scores for all movies
        similarity_scores = list(enumerate(cosine_sim[movie_idx]))
        # Sort movies by similarity score in descending order
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        # Get movie titles for the top matches
        recommended_movies = [movies.iloc[i[0]]['title'] for i in sorted_scores]
        return recommended_movies
    except IndexError:
        return "Movie not found in dataset!"
    except Exception as e:
        return str(e)

# Step 6: Test the Recommendation System
if __name__ == "__main__":
    movie = input("Enter a movie title: ")
    recommendations = recommend(movie)
    if isinstance(recommendations, list):
        print(f"\nRecommendations for '{movie}':")
        for rec in recommendations:
            print(rec)
    else:
        print(recommendations)
