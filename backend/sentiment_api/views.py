import sys
import os
import json
import random
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Adjust sys.path to include the parent folder that contains model_training.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the sentiment pipeline initialization function
from model_training import initialize_sentiment_pipeline

# Create a logger instance
logger = logging.getLogger(__name__)

# ====== Global Initialization of the model ======
try:
    sentiment_analyzer, tokenizer = initialize_sentiment_pipeline()
except Exception as e:
    sentiment_analyzer, tokenizer = None, None
    logger.error("Failed to initialize sentiment pipeline at server start: %s", e)

@csrf_exempt
def predict_sentiment(request):
    if request.method == "POST":
        if not sentiment_analyzer or not tokenizer:
            response = JsonResponse({"error": "Model not available."}, status=500)
            response["Access-Control-Allow-Origin"] = "*"
            return response
        try:
            data = json.loads(request.body.decode('utf-8'))
            review = data.get("review", "")
            if not review.strip():
                response = JsonResponse({"error": "No review provided."}, status=400)
                response["Access-Control-Allow-Origin"] = "*"
                return response
            
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
        except UnicodeDecodeError:
            response = JsonResponse({"error": "Invalid characters in review text."}, status=400)
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

# ===================== Random Image Serving API =====================

@csrf_exempt
def get_random_image(request):
    if request.method == "GET":
        try:
            sample_folder = os.path.join(settings.BASE_DIR, "sample-toys")  # ⬅️ Correct path
            images = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            if not images:
                return JsonResponse({"error": "No images found."}, status=404)

            random_image = random.choice(images)
            image_url = f"/sample-toys/{random_image}"  # relative URL for serving via static files

            response = JsonResponse({"image_url": image_url})
            response["Access-Control-Allow-Origin"] = "*"
            return response
        except Exception as e:
            logger.error("Random image error: %s", e)
            response = JsonResponse({"error": str(e)}, status=500)
            response["Access-Control-Allow-Origin"] = "*"
            return response
    else:
        response = JsonResponse({"error": "Method not allowed."}, status=405)
        response["Access-Control-Allow-Origin"] = "*"
        return response
