import pandas as pd
import logging
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

def create_custom_topic_model():
    """
    Create a BERTopic model instance with a custom vectorizer.
    
    Adjust the n-gram range and stop words to improve topic quality.
    
    Returns:
        BERTopic: A customized BERTopic model.
    """
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    # min_topic_size can be tuned for your dataset's characteristics
    topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=10)
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

def process_sentiment_analysis(reviews, sentiment_analyzer, tokenizer, max_reviews=None):
    """
    Process sentiment analysis over the provided reviews. Reviews are tokenized and
    truncated to a maximum length for processing.

    Parameters:
        reviews (List[str]): List of review texts.
        sentiment_analyzer: The pre-initialized sentiment analysis pipeline.
        tokenizer: The associated tokenizer.
        max_reviews (int, optional): Limit number of reviews for processing.
    
    Returns:
        pd.DataFrame: DataFrame containing reviews with sentiment predictions.
    """
    if max_reviews:
        reviews = reviews[:max_reviews]

    truncated_reviews = [
        tokenizer.decode(tokenizer.encode(review, max_length=512, truncation=True), skip_special_tokens=True)
        for review in reviews
    ]
    
    results = sentiment_analyzer(truncated_reviews)
    
    # Convert results into a DataFrame
    sentiment_df = pd.DataFrame(results)
    sentiment_df['review'] = truncated_reviews
    
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
    file_path = "amazon_co-ecommerce_sample.csv"
    logger.info("Loading reviews from file.")
    reviews = load_reviews(file_path)
    logger.info(f"Loaded {len(reviews)} reviews.")
    
    # Train the hierarchical topic model
    topic_model, topics, probs = train_topic_model(reviews)
    hierarchical_topics = extract_hierarchical_topics(topic_model, reviews)
    
    # Save the topic model for future predictions
    save_topic_model(topic_model, path="bertopic_model")
    
    # Initialize the sentiment analysis pipeline
    sentiment_analyzer, tokenizer = initialize_sentiment_pipeline()
    sentiment_df = process_sentiment_analysis(reviews, sentiment_analyzer, tokenizer, max_reviews=100)
    
    # Save sentiment analysis results for reference
    sentiment_df.to_csv("sentiment_results.csv", index=False)
    logger.info("Sentiment analysis results saved to 'sentiment_results.csv'.")
    
    # Optional: Visualize sentiment distribution for exploratory analysis
    visualize_sentiment(sentiment_df)

if __name__ == "__main__":
    main()
