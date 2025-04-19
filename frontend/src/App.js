import React, { useState } from 'react';

function App() {
  // State to hold the user review, the prediction result, and potential error messages.
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Function to handle form submission.
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/sentiment/", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review })
      });      
      if (!response.ok) {
        // If there is any error, throw an error to be caught in catch block.
        throw new Error('Server error: ' + response.statusText);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Sentiment Analyzer</h1>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '10px' }}>
          <label htmlFor="review" style={{ display: 'block', marginBottom: '5px' }}>Enter your review:</label>
          <textarea
            id="review"
            value={review}
            onChange={(e) => setReview(e.target.value)}
            rows="4"
            style={{ width: '100%', padding: '10px' }}
            placeholder="Type your review here..."
          />
        </div>
        <button type="submit" style={{ padding: '10px 20px' }} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: '20px', color: 'red' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '20px' }}>
          <h2>Prediction Result:</h2>
          <p>
            <strong>Sentiment:</strong> {result.label}
          </p>
          <p>
            <strong>Score:</strong> {result.score.toFixed(2)}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
