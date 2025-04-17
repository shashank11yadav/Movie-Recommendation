import pandas as pd
import numpy as np
import pickle
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk
import os

# Download NLTK data
nltk.download('punkt', quiet=True)

class MovieRecommender:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        self.ps = PorterStemmer()
        self.new_df = None
        self.similarity = None
        self.movies = None
        self.cv = None
        
        # Try to load pre-trained model, or build it if not available
        self.load_or_build_model()
    
    def load_or_build_model(self):
        try:
            # Try to load pre-trained model
            self.new_df = pickle.load(open(os.path.join(self.model_path, 'movies_df.pkl'), 'rb'))
            self.similarity = pickle.load(open(os.path.join(self.model_path, 'similarity.pkl'), 'rb'))
            self.cv = pickle.load(open(os.path.join(self.model_path, 'cv.pkl'), 'rb'))
            print("Loaded pre-trained model")
        except (FileNotFoundError, EOFError):
            print("Building model from scratch")
            self.build_model()
    
    def build_model(self):
        # Make sure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load data
        try:
            credits = pd.read_csv('tmdb_5000_credits.csv')
            movies = pd.read_csv('tmdb_5000_movies.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Dataset files not found. Please make sure tmdb_5000_credits.csv and tmdb_5000_movies.csv are in the correct location")
        
        # Merge datasets
        movies = movies.merge(credits, on='title')
        
        # Select relevant columns
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        
        # Drop rows with missing values
        movies.dropna(inplace=True)
        
        # Drop duplicates
        movies.drop_duplicates(inplace=True)
        
        # Process features
        movies['genres'] = movies['genres'].apply(self.convert)
        movies['keywords'] = movies['keywords'].apply(self.convert)
        movies['cast'] = movies['cast'].apply(self.convert3)
        movies['crew'] = movies['crew'].apply(self.fetch_director)
        movies['overview'] = movies['overview'].apply(lambda x: x.split())
        
        # Remove spaces from feature values
        movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
        
        # Combine features
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        
        # Create new dataframe with necessary columns
        self.new_df = movies[['movie_id', 'title', 'tags']]
        
        # Join all tags and convert to lowercase
        self.new_df['tags'] = self.new_df['tags'].apply(lambda x: " ".join(x))
        self.new_df['tags'] = self.new_df['tags'].apply(lambda x: x.lower())
        
        # Apply stemming
        self.new_df['tags'] = self.new_df['tags'].apply(self.stem)
        
        # Vectorize
        self.cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = self.cv.fit_transform(self.new_df['tags']).toarray()
        
        # Calculate similarity
        self.similarity = cosine_similarity(vectors)
        
        # Save model for future use
        pickle.dump(self.new_df, open(os.path.join(self.model_path, 'movies_df.pkl'), 'wb'))
        pickle.dump(self.similarity, open(os.path.join(self.model_path, 'similarity.pkl'), 'wb'))
        pickle.dump(self.cv, open(os.path.join(self.model_path, 'cv.pkl'), 'wb'))
    
    def convert(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert3(self, obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    def fetch_director(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    def stem(self, text):
        y = []
        for i in text.split():
            y.append(self.ps.stem(i))
        return " ".join(y)
    
    def recommend(self, movie):
        try:
            movie_index = self.new_df[self.new_df['title'] == movie].index[0]
            distances = self.similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            
            recommendations = []
            for i in movies_list:
                recommendations.append({
                    'title': self.new_df.iloc[i[0]].title,
                    'similarity': float(i[1])
                })
            return recommendations
        except (IndexError, KeyError):
            return []
    
    def get_all_movie_titles(self):
        return self.new_df['title'].tolist() 