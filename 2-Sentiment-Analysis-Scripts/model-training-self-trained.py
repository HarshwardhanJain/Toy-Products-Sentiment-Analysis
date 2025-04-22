import pandas as pd
import logging
import os
import re
import concurrent.futures
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.feature_extraction.text import CountVectorizer
from datasets import Dataset
import plotly.express as px

# Configure Logger
def configure_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

logger = configure_logger()

# Get new non-conflicting path
def get_iterative_path(base_path: str, extension: str = ""):
    new_path = base_path + extension
    if not os.path.exists(new_path):
        return new_path
    counter = 1
    while os.path.exists(f"{base_path}_{counter}{extension}"):
        counter += 1
    return f"{base_path}_{counter}{extension}"

# Load Reviews
def load_reviews(file_path: str, column_name: str = "customer_reviews"):
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' column missing in dataset!")
    return df[column_name].dropna().tolist()

# Preprocess Reviews
def preprocess_reviews(reviews):
    processed = []
    for review in reviews:
        review = review.lower()
        review = re.sub(r"[^a-z0-9\s.,!?']", "", review)
        processed.append(review.strip())
    return processed

# Split customer reviews based on '|' separator
def split_customer_reviews(reviews):
    split_reviews = []
    for review in reviews:
        if '|' in review:
            split_reviews.extend([r.strip() for r in review.split('|') if r.strip()])
        else:
            split_reviews.append(review)
    return split_reviews

# Label reviews based on keywords
def label_reviews_from_rating(reviews):
    labeled_data = []
    for review in reviews:
        if any(word in review.lower() for word in ["5.0", "excellent", "amazing", "great", "love", "perfect"]):
            label = 2  # Positive
        elif any(word in review.lower() for word in ["1.0", "terrible", "bad", "worst", "awful"]):
            label = 0  # Negative
        else:
            label = 1  # Neutral
        labeled_data.append((review, label))
    return labeled_data

# Create Dataset for Huggingface Trainer
def create_dataset(labeled_data, tokenizer):
    texts, labels = zip(*labeled_data)
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=512)
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    })
    return dataset

# Fine-tune a model
def train_custom_sentiment_model(labeled_data):
    logger.info("Training custom sentiment model...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    dataset = create_dataset(labeled_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="./temp_output",
        evaluation_strategy="no",
        per_device_train_batch_size=16,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        remove_unused_columns=False,
        logging_dir="./logs",
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    return model, tokenizer

# Predict Sentiments
def predict_with_custom_model(reviews, model, tokenizer, batch_size=10, max_workers=4):
    model.eval()
    device = next(model.parameters()).device
    results = []

    def predict_batch(batch):
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        return preds.tolist()

    batches = [reviews[i:i+batch_size] for i in range(0, len(reviews), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(predict_batch, batch): batch for batch in batches}
        for future in concurrent.futures.as_completed(future_to_batch):
            preds = future.result()
            results.extend(preds)

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    labeled_results = [label_map[pred] for pred in results]
    return labeled_results

# Visualize Sentiment
def visualize_sentiment(sentiment_df):
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

# MAIN
def main():
    logger.info("Loading data...")
    reviews = load_reviews(r"1-Database/amazon_co-ecommerce_sample.csv")
    reviews = preprocess_reviews(reviews)
    reviews = split_customer_reviews(reviews)

    labeled_data = label_reviews_from_rating(reviews)

    model, tokenizer = train_custom_sentiment_model(labeled_data)

    # Save Model
    model_save_path = get_iterative_path(r"3_2-Saved-Models/custom_sentiment_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Custom model saved to {model_save_path}")

    logger.info("Predicting sentiments...")
    predictions = predict_with_custom_model(reviews, model, tokenizer)

    # Save results
    sentiment_df = pd.DataFrame({"review": reviews, "label": predictions})
    csv_save_path = get_iterative_path(r"4-Results-Data/sentiment_results", extension=".csv")
    sentiment_df.to_csv(csv_save_path, index=False)
    logger.info(f"Sentiment results saved to {csv_save_path}")

    visualize_sentiment(sentiment_df)

if __name__ == "__main__":
    main()
