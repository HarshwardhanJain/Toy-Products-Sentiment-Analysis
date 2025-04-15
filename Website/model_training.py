import pandas as pd
import logging
import os
import re
import concurrent.futures
from bertopic import BERTopic
from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

def configure_logger():
    """
    Configure and return a logger for tracking execution.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def get_iterative_path(base_path: str, extension: str = ""):
    """
    Returns a new file/directory path by appending an increasing counter to the base name
    if a file/folder already exists.
    
    Parameters:
        base_path (str): Base path without counter.
        extension (str): Extension (e.g., ".csv") if needed.
    
    Returns:
        str: A new file/directory path that does not exist.
    """
    new_path = base_path + extension
    if not os.path.exists(new_path):
        return new_path
    counter = 1
    while os.path.exists(f"{base_path}_{counter}{extension}"):
        counter += 1
    return f"{base_path}_{counter}{extension}"

def load_reviews(file_path: str, column_name: str = "customer_reviews"):
    """
    Load reviews from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        column_name (str): Column name containing review texts.

    Returns:
        List[str]: A list of reviews.
        
    Raises:
        ValueError: When the expected column is not found in the dataset.
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            raise ValueError(f"Dataset does not contain a '{column_name}' column!")
        reviews = df[column_name].dropna().tolist()
        return reviews
    except Exception as e:
        logger.error(f"Error loading reviews: {e}")
        raise

def preprocess_reviews(reviews):
    """
    Preprocess the review texts by converting to lowercase, removing
    unnecessary whitespace and special characters.
    
    Parameters:
        reviews (List[str]): Original review texts.
    
    Returns:
        List[str]: Cleaned review texts.
    """
    processed = []
    for review in reviews:
        # Convert to lowercase
        review = review.lower()
        # Remove special characters (allow basic punctuation)
        review = re.sub(r"[^a-z0-9\s.,!?']", "", review)
        review = review.strip()
        processed.append(review)
    return processed

def create_custom_topic_model(min_topic_size: int = 10):
    """
    Create a BERTopic model instance with a custom vectorizer.
    
    The vectorizer uses an n-gram range of (1, 2) and English stop words.
    Further parameter tuning (e.g., stop word lists, n-gram ranges, etc.) 
    can be done based on the dataset characteristics.
    
    Parameters:
        min_topic_size (int): Minimum size of topics.
        
    Returns:
        BERTopic: A customized BERTopic model.
    """
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=min_topic_size)
    return topic_model

def train_topic_model(reviews):
    """
    Train a BERTopic model on the input reviews.

    Parameters:
        reviews (List[str]): List of customer review texts.
    
    Returns:
        topic_model (BERTopic): The trained BERTopic model.
        topics (List[int]): List of topic assignments for each review.
        probs (List[float]): Probabilities associated with topic assignments.
    """
    logger.info("Starting topic modeling training.")
    topic_model = create_custom_topic_model()
    topics, probs = topic_model.fit_transform(reviews)
    logger.info("Topic modeling training completed.")
    return topic_model, topics, probs

def extract_hierarchical_topics(topic_model, reviews):
    """
    Build a hierarchical structure of topics from the trained model.
    
    Parameters:
        topic_model (BERTopic): The trained BERTopic model.
        reviews (List[str]): List of customer review texts.
    
    Returns:
        dict: Hierarchical structure of topics.
    """
    logger.info("Extracting hierarchical topics.")
    hierarchical_topics = topic_model.hierarchical_topics(reviews)
    return hierarchical_topics

def save_topic_model(topic_model, path="bertopic_model"):
    """
    Save the trained BERTopic model to disk.

    Parameters:
        topic_model (BERTopic): The trained BERTopic model.
        path (str): File path for saving the model.
    """
    try:
        topic_model.save(path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def initialize_sentiment_pipeline():
    """
    Initialize the sentiment analysis pipeline and tokenizer using a pre-trained RoBERTa model.
    
    Returns:
        sentiment_analyzer: The sentiment analysis pipeline.
        tokenizer: The associated tokenizer.
    """
    logger.info("Initializing sentiment analysis pipeline.")
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return sentiment_analyzer, tokenizer

def process_sentiment_analysis(reviews, sentiment_analyzer, tokenizer, max_reviews=None, batch_size=10, max_workers=4):
    """
    Process sentiment analysis over the provided reviews in batches using parallel processing.
    Reviews are tokenized and truncated to a maximum length for processing.

    Parameters:
        reviews (List[str]): List of review texts.
        sentiment_analyzer: The pre-initialized sentiment analysis pipeline.
        tokenizer: The associated tokenizer.
        max_reviews (int, optional): Limit number of reviews for processing.
        batch_size (int): Number of reviews to process in each batch.
        max_workers (int): Maximum number of threads for parallel processing.
    
    Returns:
        pd.DataFrame: DataFrame containing reviews with sentiment predictions.
    """
    if max_reviews is not None:
        reviews = reviews[:max_reviews]
    
    # Helper function to process a single batch
    def process_batch(batch):
        truncated_batch = [
            tokenizer.decode(tokenizer.encode(review, max_length=512, truncation=True), skip_special_tokens=True)
            for review in batch
        ]
        batch_results = sentiment_analyzer(truncated_batch, batch_size=len(truncated_batch))
        return batch_results, truncated_batch

    batches = [reviews[i:i+batch_size] for i in range(0, len(reviews), batch_size)]
    
    all_results = []
    all_truncated_reviews = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_results, truncated_batch = future.result()
            all_results.extend(batch_results)
            all_truncated_reviews.extend(truncated_batch)
            
    # Convert results into a DataFrame
    sentiment_df = pd.DataFrame(all_results)
    sentiment_df['review'] = all_truncated_reviews
    
    # Map model labels to human-readable names
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    sentiment_df['label'] = sentiment_df['label'].map(label_map)
    
    return sentiment_df

def visualize_sentiment(sentiment_df):
    """
    Visualize the sentiment distribution using a Plotly bar chart.
    
    Parameters:
        sentiment_df (pd.DataFrame): DataFrame containing sentiment predictions.
    """
    sentiment_counts = sentiment_df['label'].value_counts().reset_index()
    sentiment_counts.columns = ['label', 'count']
    fig = px.bar(
        sentiment_counts,
        x="label",
        y="count",
        color="label",
        text="count",
        title="Sentiment Distribution"
    )
    fig.update_traces(textposition='outside', textfont_size=14)
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Frequency", title_font_size=20, template="plotly_dark")
    fig.show()

def main():
    # Initialize the logger
    global logger
    logger = configure_logger()
    
    # Load the reviews from the dataset
    file_path = r"1-Database\amazon_co-ecommerce_sample.csv"
    logger.info("Loading reviews from file.")
    reviews = load_reviews(file_path)
    logger.info(f"Loaded {len(reviews)} reviews.")
    
    # Preprocess reviews (e.g., lower-casing, cleaning text)
    reviews = preprocess_reviews(reviews)
    
    # Train the hierarchical topic model
    topic_model, topics, probs = train_topic_model(reviews)
    hierarchical_topics = extract_hierarchical_topics(topic_model, reviews)
    
    # Save the topic model with iterative naming if previous versions exist
    model_save_path = get_iterative_path(r"3-Saved-Model\bertopic_model")
    save_topic_model(topic_model, path=model_save_path)
    
    # Initialize the sentiment analysis pipeline
    sentiment_analyzer, tokenizer = initialize_sentiment_pipeline()
    
    # Process sentiment analysis using parallel processing
    sentiment_df = process_sentiment_analysis(
        reviews, 
        sentiment_analyzer, 
        tokenizer, 
        max_reviews=None, 
        batch_size=10, 
        max_workers=4
    )
    
    # Save sentiment analysis results with iterative naming under a dedicated folder
    csv_save_path = get_iterative_path(r"4-Results-Data\sentiment_results", extension=".csv")
    sentiment_df.to_csv(csv_save_path, index=False)
    logger.info(f"Sentiment analysis results saved to '{csv_save_path}'.")
    
    # Optional: Visualize sentiment distribution for exploratory analysis
    visualize_sentiment(sentiment_df)

if __name__ == "__main__":
    main()
