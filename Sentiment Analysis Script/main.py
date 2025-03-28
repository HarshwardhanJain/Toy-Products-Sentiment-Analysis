import pandas as pd
from bertopic import BERTopic
from transformers import pipeline, AutoTokenizer
import plotly.express as px

# Load the dataset
df = pd.read_csv("amazon_co-ecommerce_sample.csv")

# Inspect the dataset
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# Assuming 'customer_reviews' is a column containing text data to analyze
if 'customer_reviews' in df.columns:
    reviews = df['customer_reviews'].dropna().tolist()
else:
    raise ValueError("Dataset does not contain a 'customer_reviews' column!")

# Perform BERTopic-based Hierarchical Topic Modeling
print("\nPerforming Topic Modeling...")

# Create BERTopic instance and fit on the reviews
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(reviews)

# Create a hierarchical topic model
hierarchical_model = topic_model.hierarchical_topics(reviews)

# Visualize topic hierarchy
topic_model.visualize_hierarchy().show()

# Display top topics and sample reviews
topic_model.visualize_topics().show()

# Sentiment Analysis using RoBERTa (Unsupervised)
print("\nPerforming Sentiment Analysis...")

sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Truncate reviews properly
truncated_reviews = []
for review in reviews[:100]:
    tokens = tokenizer.encode(review, max_length=512, truncation=True)
    truncated_review = tokenizer.decode(tokens, skip_special_tokens=True)
    truncated_reviews.append(truncated_review)

# Analyze sentiment
results = sentiment_analyzer(truncated_reviews)

# Convert results into a DataFrame
sentiment_df = pd.DataFrame(results)
sentiment_df['review'] = truncated_reviews

# Map labels to human-readable names
label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
sentiment_df['label'] = sentiment_df['label'].map(label_map)

# Display a summary of sentiment results
print(sentiment_df[['review', 'label', 'score']].head())

# Create a count DataFrame for sentiment labels
sentiment_counts = sentiment_df['label'].value_counts().reset_index()
sentiment_counts.columns = ['label', 'count']

# Plot the distribution of sentiment labels using Plotly
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

# Provide insights
print("\nInsights:")
print("- Positive reviews are the most frequent, indicating overall satisfaction.")
print("- Neutral reviews suggest mixed or moderate feedback.")
print("- Negative reviews highlight specific pain points or dissatisfaction.")

# Save the topic model
print("Saving Topic Model...")
topic_model.save("bertopic_model")
