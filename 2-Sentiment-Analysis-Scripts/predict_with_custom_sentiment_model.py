import os
import re
import random
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Reload model and tokenizer
model_path = r"3_2-Saved-Models/custom_sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# Load reviews with explicit encoding handling
csv_path = r"1-Database/amazon_co-ecommerce_sample.csv"
df = pd.read_csv(csv_path, encoding='utf-8')
reviews = df["customer_reviews"].dropna().tolist()

# Preprocessing

def preprocess_reviews(reviews):
    processed = []
    for review in reviews:
        if not isinstance(review, str):
            continue
        review = review.lower()
        review = re.sub(r"[^a-z0-9\s.,!?']", "", review)  # Corrected regex
        processed.append(review.strip())
    return processed

def split_customer_reviews(reviews):
    split_reviews = []
    for review in reviews:
        if '|' in review:
            split_reviews.extend([r.strip() for r in review.split('|') if r.strip()])
        else:
            split_reviews.append(review)
    return split_reviews

# Apply preprocessing
reviews = preprocess_reviews(reviews)
reviews = split_customer_reviews(reviews)

# Shuffle reviews to avoid batch correlation bias
random.shuffle(reviews)

# Predict function

def predict_sentiment(reviews, model, tokenizer, batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.to(device)
    except Exception as e:
        print(f"Warning: Failed to move model to CUDA. Falling back to CPU. Error: {e}")
        device = torch.device("cpu")
        model.to(device)

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    results = []

    for i in tqdm(range(0, len(reviews), batch_size), desc="Predicting Sentiments"):
        batch = reviews[i:i+batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            results.extend([label_map[p.item()] for p in preds])
        except Exception as e:
            print(f"Error during batch {i//batch_size}: {e}. Skipping batch.")
            continue

    return results

# Perform predictions
predictions = predict_sentiment(reviews, model, tokenizer)

# Save results with versioned filename to avoid overwriting
save_dir = "4_2-Results-Data"
os.makedirs(save_dir, exist_ok=True)

base_filename = "sentiment_results"
save_path = os.path.join(save_dir, f"{base_filename}.csv")

if os.path.exists(save_path):
    counter = 1
    while os.path.exists(os.path.join(save_dir, f"{base_filename}_{counter}.csv")):
        counter += 1
    save_path = os.path.join(save_dir, f"{base_filename}_{counter}.csv")

# Save the dataframe
sentiment_df = pd.DataFrame({"review": reviews, "label": predictions})
sentiment_df.to_csv(save_path, index=False, encoding='utf-8')

print(f"Saved predictions to: {save_path}")
