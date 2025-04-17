import { useState, useEffect } from 'react'
import MovieSelector from './components/MovieSelector'
import MovieRecommendations from './components/MovieRecommendations'
import './App.css'
import axios from 'axios'

function App() {
  const [movies, setMovies] = useState([])
  const [selectedMovie, setSelectedMovie] = useState(null)
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    // Fetch the list of movies when component mounts
    const fetchMovies = async () => {
      try {
        const response = await axios.get('http://localhost:5001/api/movies')
        if (response.data.success) {
          setMovies(response.data.movies)
        } else {
          setError('Failed to fetch movies')
        }
      } catch (err) {
        setError('Error connecting to server')
        console.error(err)
      }
    }

    fetchMovies()
  }, [])

  const handleMovieSelect = (movie) => {
    setSelectedMovie(movie)
    if (movie) {
      getRecommendations(movie.value)
    } else {
      setRecommendations([])
    }
  }

  const getRecommendations = async (movieTitle) => {
    setLoading(true)
    setError('')
    try {
      const response = await axios.post('http://localhost:5001/api/recommend', {
        movie: movieTitle
      })
      
      if (response.data.success) {
        setRecommendations(response.data.recommendations)
      } else {
        setError(response.data.error || 'Failed to get recommendations')
        setRecommendations([])
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error getting recommendations')
      setRecommendations([])
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <h1>MovieMatch</h1>
        <p>Discover films tailored to your taste</p>
      </header>

      <main>
        <MovieSelector 
          movies={movies} 
          selectedMovie={selectedMovie} 
          onMovieSelect={handleMovieSelect} 
        />
        
        <MovieRecommendations 
          recommendations={recommendations} 
          loading={loading} 
          error={error} 
        />
      </main>

      <footer>
        <p>MovieMatch © {new Date().getFullYear()} | Film Recommendation System</p>
      </footer>
    </div>
  )
}

export default App
