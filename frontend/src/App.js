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
  const [analysisTime, setAnalysisTime] = useState(null);

  const [images, setImages] = useState([]);  // Store all images
  const [currentImageIndex, setCurrentImageIndex] = useState(0);  // Current index

  // Fetch all images once
  const fetchAllImages = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/all-images/");
      const data = await response.json();
      if (response.ok && data.images) {
        setImages(data.images);
        setCurrentImageIndex(0); // Start at first image
      } else {
        console.error("Failed to load images");
      }
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  };

  useEffect(() => {
    document.title = '🔍 Sentiment Analyzer';
    fetchAllImages();
  }, []);

  const handlePrevImage = () => {
    setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  const handleNextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % images.length);
  };

  const countWords = (text) => text.trim().split(/\s+/).filter(Boolean).length;
  const countChars = (text) => text.length;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (countWords(review) < 5) {
      toast.error("🚫 Please enter at least 5 words for better analysis.");
      return;
    }

    setLoading(true);
    const startTime = performance.now();

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/sentiment/", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review }),
      });

      if (!response.ok) {
        throw new Error('Server error: ' + response.statusText);
      }

      const data = await response.json();
      const endTime = performance.now();
      setAnalysisTime(((endTime - startTime) / 1000).toFixed(2));

      setResult(data);
      setReview('');
      toast.success("🎉 Sentiment analysis complete!");
    } catch (err) {
      setError(err.message);
      toast.error(`❌ ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setReview('');
    setResult(null);
    setError(null);
    setAnalysisTime(null);
    toast.info("🧹 Cleared review and result!");
  };

  const getBackgroundColor = () => {
    if (!result) return 'bg-white';
    if (result.label === 'Positive') return 'bg-green-100';
    if (result.label === 'Negative') return 'bg-red-100';
    if (result.label === 'Neutral') return 'bg-yellow-100';
    return 'bg-white';
  };

  const getSentimentEmoji = (label) => {
    if (label === 'Positive') return '😃';
    if (label === 'Negative') return '😞';
    if (label === 'Neutral') return '😐';
    return '';
  };

  return (
    <div className={`min-h-screen flex flex-col md:flex-row items-center justify-center ${getBackgroundColor()} transition-colors duration-700`}>
      
      {/* Left: Form Section */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
        className="w-full md:w-1/2 p-8 bg-white rounded-2xl shadow-2xl m-4"
      >
        <h1 className="text-4xl font-bold mb-6 text-center text-gray-800 tracking-wide">🔍 Sentiment Analyzer</h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="review" className="block text-lg font-semibold mb-2 text-gray-700">
              Enter your review:
            </label>
            <textarea
              id="review"
              value={review}
              onChange={(e) => setReview(e.target.value)}
              rows="5"
              placeholder="Type something like 'The product quality is amazing!'..."
              autoFocus
              className="w-full p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none text-gray-700 placeholder:italic placeholder:text-gray-400"
            />
            <div className="mt-2 text-sm text-gray-500">
              Words: {countWords(review)} | Characters: {countChars(review)}
            </div>
          </div>

          <div className="flex justify-center space-x-4">
            <button
              type="submit"
              className={`font-semibold py-2 px-6 rounded-lg flex items-center justify-center transition
              ${loading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
              disabled={loading}
            >
              {loading ? (
                <Circles height="24" width="24" color="white" ariaLabel="circles-loading" visible={true} />
              ) : (
                "Analyze"
              )}
            </button>

            <button
              type="button"
              onClick={handleClear}
              className={`font-semibold py-2 px-6 rounded-lg transition
              ${loading || review.trim().length === 0 ? 'bg-gray-400 cursor-not-allowed' : 'bg-gray-500 hover:bg-gray-600 text-white'}`}
              disabled={loading || review.trim().length === 0}
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
            transition={{ delay: 0.3 }}
            className="mt-8 text-center"
          >
            <h2 className="text-2xl font-bold mb-4 text-gray-800">📈 Prediction Result</h2>
            <p className="text-xl text-gray-700">
              <strong>Sentiment:</strong> {result.label} {getSentimentEmoji(result.label)}
            </p>
            <div className="w-full mt-4">
              <div className="w-full bg-gray-200 rounded-full h-4">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(result.score * 100).toFixed(0)}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className="bg-blue-600 h-4 rounded-full"
                />
              </div>
              <p className="mt-2 text-gray-600 text-sm">Confidence: {(result.score * 100).toFixed(2)}%</p>
              {analysisTime && (
                <p className="mt-1 text-gray-500 text-sm">Analyzed in {analysisTime} sec</p>
              )}
            </div>
          </motion.div>
        )}

        <ToastContainer
          position="bottom-right"
          autoClose={3000}
          hideProgressBar={false}
          newestOnTop
          closeOnClick
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="colored"
        />
      </motion.div>

      {/* Right: Product Image Section */}
      <motion.div
        initial={{ opacity: 0, x: 100 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.7 }}
        className="hidden md:flex w-full md:w-1/2 flex-col items-center p-6 bg-white rounded-2xl shadow-2xl m-4 relative"
      >
        <h2 className="text-2xl font-bold mb-4 text-gray-800">🛒 Product Reference</h2>
        
        {/* Left Arrow */}
        <button
          onClick={handlePrevImage}
          className="absolute left-4 top-1/2 transform -translate-y-1/2 text-3xl font-bold text-gray-500 hover:text-gray-700"
        >
          ←
        </button>

        {/* Image */}
        {images.length > 0 && (
          <img
            src={`http://127.0.0.1:8000${images[currentImageIndex]}`}
            alt="Sample Product"
            className="rounded-xl shadow-md object-cover w-80 h-80"
          />
        )}

        {/* Right Arrow */}
        <button
          onClick={handleNextImage}
          className="absolute right-4 top-1/2 transform -translate-y-1/2 text-3xl font-bold text-gray-500 hover:text-gray-700"
        >
          →
        </button>

        <p className="mt-4 text-gray-500 text-sm text-center">
          (Select product visual to guide your review.)
        </p>
      </motion.div>

    </div>
  );
}

export default App;
