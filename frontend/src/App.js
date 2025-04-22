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
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [compareMode, setCompareMode] = useState(false);

  useEffect(() => {
    document.title = '🔍 Sentiment Analyzer';
    fetchAllImages();
  }, []);

  const fetchAllImages = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/all-images/");
      const data = await response.json();
      if (response.ok && data.images) {
        setImages(data.images);
      } else {
        console.error("Failed to load images");
      }
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  };

  const handlePrevImage = () => {
    setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  const handleNextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % images.length);
  };

  const handleSurpriseMe = () => {
    if (images.length > 1) {
      let randomIndex;
      do {
        randomIndex = Math.floor(Math.random() * images.length);
      } while (randomIndex === currentImageIndex);
      setCurrentImageIndex(randomIndex);
    }
  };

  const countWords = (text) => text.trim().split(/\s+/).length;
  const countChars = (text) => text.length;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (countWords(review) < 5) {
      toast.error("🚫 Please enter at least 5 words for better analysis.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const startTime = performance.now();

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/sentiment/", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          review,
          model: useCustomModel ? 'custom' : 'pretrained',
          compare: compareMode,
        }),
      });

      if (!response.ok) throw new Error('Server error: ' + response.statusText);

      const data = await response.json();
      const endTime = performance.now();

      setResult(data);
      setAnalysisTime(((endTime - startTime) / 1000).toFixed(2));
      toast.success("🎉 Sentiment analysis complete!");
    } catch (err) {
      toast.error(`❌ ${err.message}`);
      setError(err.message);
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
    if (compareMode) return 'bg-blue-50';
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

      {/* Left Section */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
        className="w-full md:w-1/2 p-8 bg-white rounded-2xl shadow-2xl m-4"
      >
        <h1 className="text-4xl font-bold mb-6 text-center text-gray-800">🔍 Sentiment Analyzer</h1>

        {/* Toggle Switches */}
        <div className="flex items-center justify-center mb-6 space-x-6">
          {/* Use Custom Model Toggle */}
          <div className="flex items-center">
            <span className="mr-2 font-semibold text-gray-700">Use Custom Model</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={useCustomModel}
                onChange={() => {
                  if (!compareMode) setUseCustomModel(!useCustomModel);
                }}
                disabled={compareMode}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          {/* Compare Mode Toggle */}
          <div className="flex items-center">
            <span className="mr-2 font-semibold text-gray-700">Compare Both</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" checked={compareMode} onChange={() => setCompareMode(!compareMode)} />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-purple-500 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
            </label>
          </div>
        </div>

        {/* Form */}
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
              className="w-full p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none text-gray-700"
            />
            <div className="mt-2 text-sm text-gray-500">
              Words: {countWords(review)} | Characters: {countChars(review)}
            </div>
          </div>

          <div className="flex justify-center space-x-4">
            <button
              type="submit"
              disabled={loading}
              className={`font-semibold py-2 px-6 rounded-lg transition ${loading ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
            >
              {loading ? (
                <Circles height="24" width="24" color="white" ariaLabel="circles-loading" visible={true} />
              ) : (
                "Analyze"
              )}
            </button>

            <button
              type="button"
              disabled={loading || review.trim().length === 0}
              onClick={handleClear}
              className="font-semibold py-2 px-6 rounded-lg bg-gray-500 hover:bg-gray-600 text-white transition"
            >
              Clear
            </button>
          </div>
        </form>

        {/* Results (Single or Compare) */}
        {/* Results Section */}
        {error && (
          <div className="mt-6 p-4 bg-red-100 text-red-800 rounded-lg text-center">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Single Prediction */}
        {result && !compareMode && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="mt-8 text-center">
            <h2 className="text-2xl font-bold mb-4 text-gray-800">📈 Prediction Result</h2>
            <p className="text-xl text-gray-700 mb-2">
              <strong>Sentiment:</strong> {result.label} {getSentimentEmoji(result.label)}
            </p>
            <p className="text-sm text-gray-500 mb-2">
              <strong>Model Used:</strong> {result.model_used}
            </p>
            <div className="w-full bg-gray-300 rounded-full h-4 mx-auto max-w-md mb-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(result.score * 100).toFixed(0)}%` }}
                transition={{ duration: 0.8 }}
                className="h-4 rounded-full bg-blue-600"
              />
            </div>
            <p className="text-sm text-gray-600">Confidence: {(result.score * 100).toFixed(2)}%</p>
            {analysisTime && <p className="mt-2 text-gray-500 text-sm">Analyzed in {analysisTime} sec</p>}
          </motion.div>
        )}

        {/* Compare Mode Prediction */}
        {result && compareMode && (
          <div className="mt-8 space-y-6">
            {/* Verdict */}
            <div className="text-center mb-6">
              {result.pretrained.label === result.custom.label ? (
                <p className="text-green-600 font-semibold text-lg">
                  ✅ Both models agree: <span className="capitalize">{result.pretrained.label}</span>!
                </p>
              ) : (
                <p className="text-red-600 font-semibold text-lg">
                  ⚡ Models differ: Pretrained → <span className="capitalize">{result.pretrained.label}</span>, Custom → <span className="capitalize">{result.custom.label}</span>
                </p>
              )}
            </div>

            {/* Dual Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Pretrained */}
              <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="p-6 bg-gray-100 rounded-xl shadow-md text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-3">🧠 Pretrained Model</h3>
                <p className="text-lg mb-2">
                  <strong>Sentiment:</strong> {result.pretrained.label} {getSentimentEmoji(result.pretrained.label)}
                </p>
                <div className="w-full bg-gray-300 rounded-full h-4 mx-auto max-w-md mb-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(result.pretrained.score * 100).toFixed(0)}%` }}
                    transition={{ duration: 0.8 }}
                    className="h-4 rounded-full bg-blue-600"
                  />
                </div>
                <p className="text-sm text-gray-600">Confidence: {(result.pretrained.score * 100).toFixed(2)}%</p>
                <p className="text-sm text-gray-500 mt-1">Model: {result.pretrained.model_used}</p>
              </motion.div>

              {/* Custom */}
              <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="p-6 bg-gray-100 rounded-xl shadow-md text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-3">🛠️ Custom Model</h3>
                <p className="text-lg mb-2">
                  <strong>Sentiment:</strong> {result.custom.label} {getSentimentEmoji(result.custom.label)}
                </p>
                <div className="w-full bg-gray-300 rounded-full h-4 mx-auto max-w-md mb-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(result.custom.score * 100).toFixed(0)}%` }}
                    transition={{ duration: 0.8 }}
                    className="h-4 rounded-full bg-purple-600"
                  />
                </div>
                <p className="text-sm text-gray-600">Confidence: {(result.custom.score * 100).toFixed(2)}%</p>
                <p className="text-sm text-gray-500 mt-1">Model: {result.custom.model_used}</p>
              </motion.div>
            </div>

            {/* Analyzed time */}
            {analysisTime && (
              <p className="mt-4 text-center text-gray-500 text-sm">Analyzed in {analysisTime} sec</p>
            )}
          </div>
        )}

        <ToastContainer position="bottom-right" autoClose={3000} hideProgressBar={false} theme="colored" />
      </motion.div>

      {/* Right Section — Product Images */}
      <motion.div
        initial={{ opacity: 0, x: 100 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.7 }}
        className="hidden md:flex w-full md:w-1/2 flex-col items-center p-6 bg-white rounded-2xl shadow-2xl m-4 relative"
      >
        <h2 className="text-2xl font-bold mb-4 text-gray-800">🛒 Product Reference</h2>

        {/* Left Arrow */}
        <button onClick={handlePrevImage} className="absolute left-4 top-1/2 transform -translate-y-1/2 text-3xl font-bold text-gray-500 hover:text-gray-700">
          ←
        </button>

        {/* Image */}
        {images.length > 0 && (
          <motion.img
            key={currentImageIndex}
            src={`http://127.0.0.1:8000${images[currentImageIndex]}`}
            alt="Product"
            className="rounded-xl shadow-md object-cover w-80 h-80"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          />
        )}

        {/* Right Arrow */}
        <button onClick={handleNextImage} className="absolute right-4 top-1/2 transform -translate-y-1/2 text-3xl font-bold text-gray-500 hover:text-gray-700">
          →
        </button>

        {/* Surprise Me Button */}
        <motion.button whileTap={{ scale: 1.2, y: -5 }} onClick={handleSurpriseMe} className="mt-6 px-5 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-full font-semibold shadow-md">
          🎲 Surprise Me
        </motion.button>

        <p className="mt-4 text-gray-500 text-sm text-center">
          (Select or surprise yourself with a product visual.)
        </p>
      </motion.div>
    </div>
  );
}

export default App;
