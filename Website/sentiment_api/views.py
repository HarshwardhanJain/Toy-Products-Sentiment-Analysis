import sys
import os

# Adjust sys.path to include the parent folder that contains model_training.py.
# This assumes the structure mentioned above.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Import the sentiment pipeline initialization function from model_training.py
from model_training import initialize_sentiment_pipeline

# Create a logger instance for this module.
logger = logging.getLogger(__name__)

@csrf_exempt
def predict_sentiment(request):
    """
    Processes a POST request to predict the sentiment of a review.
    Expects a JSON body: { "review": "Your review text" }
    """
    if request.method == "POST":
        try:
            # Decode the request body (bytes to string) and parse the JSON.
            data = json.loads(request.body.decode('utf-8'))
            review = data.get("review", "")
            
            if not review:
                return JsonResponse({"error": "No review provided."}, status=400)
            
            # Initialize the sentiment analysis pipeline using your saved model.
            sentiment_analyzer, tokenizer = initialize_sentiment_pipeline()
            
            # Tokenize and truncate the review text to avoid errors.
            truncated_review = tokenizer.decode(
                tokenizer.encode(review, max_length=512, truncation=True),
                skip_special_tokens=True
            )
            
            # Get the sentiment prediction.
            result = sentiment_analyzer(truncated_review)
            
            # Validate that the result is in the expected format.
            if not result or 'label' not in result[0]:
                raise ValueError("Unexpected prediction output format.")
            
            # Map model output labels to human-readable text.
            label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
            result[0]['label'] = label_map.get(result[0]['label'], result[0]['label'])
            
            return JsonResponse(result[0])
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Method not allowed."}, status=405)
