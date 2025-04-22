import pandas as pd
from bertopic import BERTopic
import plotly.express as px

print("Loading the saved BERTopic model...")
topic_model = BERTopic.load(r"3-Saved-Model\bertopic_model_1")

df = pd.read_csv(r"1-Database\amazon_co-ecommerce_sample.csv")

if 'customer_reviews' in df.columns:
    reviews = df['customer_reviews'].dropna().tolist()
else:
    raise ValueError("Dataset does not contain a 'customer_reviews' column!")

# Transform reviews using the loaded model
print("\nTransforming reviews using the loaded model...")
topics, probs = topic_model.transform(reviews)

# Visualize topic hierarchy
print("\nVisualizing topic hierarchy...")
topic_model.visualize_hierarchy().show()

# Display top topics and sample reviews
print("\nVisualizing top topics...")
topic_model.visualize_topics().show()

print("\nMost frequent topics:")
print(topic_model.get_topic_info().head())

# Sentiment Analysis
print("\nLoading sentiment analysis results...")

# Load or Re-compute sentiment analysis if needed
sentiment_df = pd.read_csv(r"4-Results-Data\sentiment_results_1.csv")
if sentiment_df.empty:
    raise ValueError("Sentiment results file not found. Re-run sentiment analysis.")

# Plot sentiment distribution
sentiment_counts = sentiment_df['label'].value_counts().reset_index()
sentiment_counts.columns = ['label', 'count']

fig = px.bar(
    sentiment_counts,
    x="label",
    y="count",
    color="label",
    text="count",
    title="Sentiment Distribution",
)
fig.update_traces(textposition='outside', textfont_size=14)
fig.update_layout(
    xaxis_title="Sentiment",
    yaxis_title="Frequency",
    title_font_size=20,
    template="plotly_dark"
)
fig.show()

print("\nSentiment Summary:")
print(sentiment_df[['review', 'label', 'score']].head())

print("\nInsights:")
print("- Positive reviews are the most frequent, indicating overall satisfaction.")
print("- Neutral reviews suggest mixed or moderate feedback.")
print("- Negative reviews highlight specific pain points or dissatisfaction.")
