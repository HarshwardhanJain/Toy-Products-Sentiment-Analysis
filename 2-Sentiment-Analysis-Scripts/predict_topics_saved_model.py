from bertopic import BERTopic

# Load the saved model
topic_model = BERTopic.load("bertopic_model")

new_reviews = [
    "I loved the product quality and the service was excellent.",
    "The packaging was poor and the item arrived damaged.",
    # Add more reviews as needed
]

topics, probs = topic_model.transform(new_reviews)
print("Predicted Topics:", topics)
print("Probabilities:", probs)
