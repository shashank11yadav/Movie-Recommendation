import { useMemo } from 'react'
import Select from 'react-select'

const MovieSelector = ({ movies, selectedMovie, onMovieSelect }) => {
  // Format movies for react-select
  const movieOptions = useMemo(() => {
    return movies.map(movie => ({
      value: movie,
      label: movie
    }))
  }, [movies])

  return (
    <div className="movie-selector">
      <h2>Choose a Movie</h2>
      <Select
        className="movie-select"
        value={selectedMovie}
        onChange={onMovieSelect}
        options={movieOptions}
        isClearable
        isSearchable
        placeholder="Search for a movie..."
        noOptionsMessage={() => "No movies found"}
      />
    </div>
  )
}

export default MovieSelector 