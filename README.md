# MovieMatch - Film Recommendation System

![MovieMatch Logo](frontend/public/favicon.svg)

A modern web application that recommends movies based on content similarity. This system uses natural language processing and machine learning techniques to find films similar to your selected movie.

## 📋 Features

- **Smart Recommendations**: Discover movies with similar content using ML algorithms
- **Sleek Interface**: Modern, responsive design that works on all devices
- **Fast Suggestions**: Get instant recommendations with similarity scores
- **Search Functionality**: Easily find movies from thousands of options

## 🛠️ Tech Stack

- **Frontend**: React with Vite, React-Select for search functionality
- **Backend**: Flask REST API with CORS support
- **ML Pipeline**: scikit-learn, NLTK for NLP processing
- **Algorithms**: TF-IDF vectorization, cosine similarity, stemming

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ for backend
- Node.js 14+ and npm for frontend
- The TMDb movie dataset files:
  - `tmdb_5000_credits.csv`
  - `tmdb_5000_movies.csv`

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the dataset files are in the backend directory.

5. Start the Flask server:
   ```bash
   python run.py
   ```
   The API will be available at http://localhost:5001.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```
   The app will be available at http://localhost:5173.

## 📱 API Documentation

### GET `/api/movies`

Returns a list of all available movies.

**Response Format**:
```json
{
  "success": true,
  "movies": ["Movie 1", "Movie 2", "..."]
}
```

### POST `/api/recommend`

Returns movie recommendations based on a provided movie title.

**Request Format**:
```json
{
  "movie": "Movie Title"
}
```

**Response Format**:
```json
{
  "success": true,
  "recommendations": [
    {
      "title": "Recommended Movie 1",
      "similarity": 0.85
    },
    {
      "title": "Recommended Movie 2",
      "similarity": 0.76
    }
  ]
}
```

## 🌐 Deployment Guide

### Backend Deployment on Render

1. Create a free account at [render.com](https://render.com)
2. Create a new Web Service and connect your GitHub repository
3. Configure the service:
   - **Name**: moviematch-api (or your preferred name)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:create_app()`
   - **Root Directory**: `backend`

4. Add your movie dataset files:
   - Go to your service dashboard
   - Navigate to the "Disks" tab
   - Create a permanent disk and mount it to `/data`
   - Upload your dataset files to this location
   - Update your code to reference `/data/tmdb_5000_credits.csv`

### Frontend Deployment on Vercel

1. Create a free account at [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Configure the project:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

4. Add environment variables:
   - `VITE_API_URL`: Your Render backend URL (e.g., `https://moviematch-api.onrender.com`)

5. Deploy your site

## 🧪 Testing

To run a complete test of the application:

1. Ensure both frontend and backend are running
2. Visit the frontend URL
3. Search for a movie (e.g., "The Dark Knight")
4. Verify that recommendations appear
5. Check that similarity scores and UI elements display correctly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

If you have any questions or feedback, please open an issue on this repository. 