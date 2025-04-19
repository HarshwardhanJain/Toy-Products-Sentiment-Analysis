import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { motion } from 'framer-motion';
import { Circles } from 'react-loader-spinner';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    if (!review.trim()) {
      toast.error("🚫 Please enter a review before submitting!");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/sentiment/", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review })
      });
      if (!response.ok) {
        throw new Error('Server error: ' + response.statusText);
      }
      const data = await response.json();
      setResult(data);
      toast.success("🎉 Sentiment analysis successful!");
    } catch (err) {
      setError(err.message);
      toast.error("❌ " + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (result) setError(null);
  }, [result]);

  const handleClear = () => {
    setReview('');
    setResult(null);
    setError(null);
    toast.info("🧹 Cleared input and result!");
  };

  const getBackgroundColor = () => {
    if (!result) return 'bg-white';
    if (result.label === 'Positive') return 'bg-green-100';
    if (result.label === 'Negative') return 'bg-red-100';
    if (result.label === 'Neutral') return 'bg-yellow-100';
    return 'bg-white';
  };

  return (
    <div className={`min-h-screen flex items-center justify-center ${getBackgroundColor()} transition-colors duration-500`}>
      <motion.div 
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-2xl p-8 bg-white rounded-lg shadow-lg"
      >
        <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">🔍 Sentiment Analyzer</h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="review" className="block text-lg font-semibold mb-2 text-gray-700">Your Review:</label>
            <textarea
              id="review"
              value={review}
              onChange={(e) => setReview(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  handleSubmit(e);
                }
              }}
              rows="5"
              placeholder="Type something like 'This product is amazing!'..."
              className="w-full p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none text-gray-700"
            />
          </div>

          <div className="flex justify-center space-x-4">
            <button 
              type="submit" 
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition"
              disabled={loading}
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <Circles
                    height="20"
                    width="20"
                    color="white"
                    ariaLabel="circles-loading"
                    visible={true}
                  />
                  <span>Analyzing...</span>
                </div>
              ) : (
                "Analyze"
              )}
            </button>

            <button 
              type="button" 
              onClick={handleClear}
              className="bg-gray-500 hover:bg-gray-600 text-white font-semibold py-2 px-6 rounded-lg transition"
              disabled={loading}
            >
              Clear
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-100 text-red-800 rounded-lg text-center">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mt-8 text-center"
          >
            <h2 className="text-2xl font-bold mb-4 text-gray-800">📈 Prediction Result</h2>
            <p className="text-xl text-gray-700">
              <strong>Sentiment:</strong> {result.label}
            </p>
            <p className="text-lg text-gray-600 mt-2">
              <strong>Confidence:</strong> {result.score.toFixed(2)}
            </p>
          </motion.div>
        )}

        <ToastContainer position="bottom-right" autoClose={3000} hideProgressBar />
      </motion.div>
    </div>
  );
}

export default App;
