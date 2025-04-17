const MovieRecommendations = ({ recommendations, loading, error }) => {
  if (loading) {
    return (
      <div className="recommendations">
        <h2>Finding Recommendations...</h2>
        <div className="loader"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="recommendations">
        <h2>Recommendations</h2>
        <div className="error-message">{error}</div>
      </div>
    )
  }

  if (recommendations.length === 0) {
    return (
      <div className="recommendations">
        <h2>Recommendations</h2>
        <p>Select a movie to see recommendations</p>
      </div>
    )
  }

  return (
    <div className="recommendations">
      <h2>Recommendations</h2>
      <div className="recommendations-list">
        {recommendations.map((recommendation, index) => (
          <div className="recommendation-card" key={index}>
            <h3>{recommendation.title}</h3>
            <div className="similarity-score">
              <p>Similarity Score: {(recommendation.similarity * 100).toFixed(2)}%</p>
              <div 
                className="similarity-bar" 
                style={{ width: `${recommendation.similarity * 100}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default MovieRecommendations 