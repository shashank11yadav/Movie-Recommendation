from flask import Blueprint, jsonify, request
from .movie_recommender import MovieRecommender

main_bp = Blueprint('main', __name__)
recommender = MovieRecommender()

@main_bp.route('/api/movies', methods=['GET'])
def get_movies():
    try:
        # Get all movie titles
        movies = recommender.get_all_movie_titles()
        return jsonify({'success': True, 'movies': movies})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/api/recommend', methods=['POST'])
def recommend_movies():
    try:
        data = request.get_json()
        movie_title = data.get('movie')
        
        if not movie_title:
            return jsonify({'success': False, 'error': 'Movie title is required'}), 400
        
        # Get recommendations
        recommendations = recommender.recommend(movie_title)
        
        if not recommendations:
            return jsonify({'success': False, 'error': f'No recommendations found for movie: {movie_title}'}), 404
        
        return jsonify({'success': True, 'recommendations': recommendations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500 