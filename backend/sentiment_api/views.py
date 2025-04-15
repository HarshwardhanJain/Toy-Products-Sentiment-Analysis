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
    if request.method == "OPTIONS":
        response = JsonResponse({})
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        response["Access-Control-Allow-Origin"] = "*"  # Development only
        return response

    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))
            review = data.get("review", "")
            if not review:
                response = JsonResponse({"error": "No review provided."}, status=400)
                response["Access-Control-Allow-Origin"] = "*"
                return response
            
            sentiment_analyzer, tokenizer = initialize_sentiment_pipeline()
            truncated_review = tokenizer.decode(
                tokenizer.encode(review, max_length=512, truncation=True),
                skip_special_tokens=True
            )
            result = sentiment_analyzer(truncated_review)
            
            if not result or 'label' not in result[0]:
                raise ValueError("Unexpected prediction output format.")

            label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
            result[0]['label'] = label_map.get(result[0]['label'], result[0]['label'])
            response = JsonResponse(result[0])
            response["Access-Control-Allow-Origin"] = "*"
            return response
        except Exception as e:
            logger.error("Prediction error: %s", e)
            response = JsonResponse({"error": str(e)}, status=500)
            response["Access-Control-Allow-Origin"] = "*"
            return response
    else:
        response = JsonResponse({"error": "Method not allowed."}, status=405)
        response["Access-Control-Allow-Origin"] = "*"
        return response
