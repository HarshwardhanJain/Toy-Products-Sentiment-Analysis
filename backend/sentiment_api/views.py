import sys
import os
import json
import logging
import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Setup logger
logger = logging.getLogger(__name__)

# Global model objects
pretrained_analyzer = None
pretrained_tokenizer = None
custom_model = None
custom_tokenizer = None
cached_custom_model_path = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
pretrained_model_path = "cardiffnlp/twitter-roberta-base-sentiment"

# Get latest custom model path
def get_latest_custom_model_path():
    global cached_custom_model_path

    if cached_custom_model_path is not None:
        return cached_custom_model_path

    base_dir = os.path.join(parent_dir, '3_2-Saved-Models')
    base_name = "custom_sentiment_model"
    
    folders = [
        f for f in os.listdir(base_dir)
        if f.startswith(base_name) and os.path.isdir(os.path.join(base_dir, f))
    ]
    
    if not folders:
        raise FileNotFoundError("No custom model folders found.")

    def extract_index(folder_name):
        parts = folder_name[len(base_name):].strip('_')
        try:
            return int(parts)
        except ValueError:
            return 0

    latest_folder = max(folders, key=extract_index)
    latest_path = os.path.join(base_dir, latest_folder)

    cached_custom_model_path = latest_path
    return latest_path

# Loaders
def load_pretrained_model():
    global pretrained_analyzer, pretrained_tokenizer
    pretrained_analyzer = pipeline("sentiment-analysis", model=pretrained_model_path)
    pretrained_tokenizer = pretrained_analyzer.tokenizer
    logger.info(f"Pretrained model loaded: {pretrained_model_path}")

def load_custom_model():
    global custom_model, custom_tokenizer
    custom_model_path = get_latest_custom_model_path()
    custom_tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
    custom_model = AutoModelForSequenceClassification.from_pretrained(custom_model_path)
    custom_model.eval()
    custom_model.to(device)
    logger.info(f"Custom model loaded: {os.path.basename(custom_model_path)}")

# Predict API
@csrf_exempt
def predict_sentiment(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        review = data.get("review", "")
        model_choice = data.get("model", "pretrained")
        compare_mode = data.get("compare", False)

        if not review.strip():
            return JsonResponse({"error": "No review provided."}, status=400)

        results = {}

        if compare_mode:
            if pretrained_analyzer is None:
                load_pretrained_model()
            if custom_model is None:
                load_custom_model()

            # Pretrained prediction
            truncated_review = pretrained_tokenizer.decode(
                pretrained_tokenizer.encode(review, max_length=512, truncation=True),
                skip_special_tokens=True
            )
            pretrained_result = pretrained_analyzer(truncated_review)[0]
            pretrained_label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

            # Custom prediction
            inputs = custom_tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = custom_model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=-1).item()
                score = torch.softmax(logits, dim=-1)[0][pred].item()
            custom_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

            results = {
                "pretrained": {
                    "label": pretrained_label_map.get(pretrained_result['label'], pretrained_result['label']),
                    "score": pretrained_result.get('score', 0),
                    "model_used": pretrained_model_path
                },
                "custom": {
                    "label": custom_label_map.get(pred, str(pred)),
                    "score": score,
                    "model_used": os.path.basename(get_latest_custom_model_path())
                }
            }

        else:
            if model_choice == "custom":
                if custom_model is None:
                    load_custom_model()

                inputs = custom_tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = custom_model(**inputs)
                    logits = outputs.logits
                    pred = torch.argmax(logits, dim=-1).item()
                    score = torch.softmax(logits, dim=-1)[0][pred].item()

                label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                results = {
                    "label": label_map.get(pred, str(pred)),
                    "score": score,
                    "model_used": os.path.basename(get_latest_custom_model_path())
                }
            else:
                if pretrained_analyzer is None:
                    load_pretrained_model()

                truncated_review = pretrained_tokenizer.decode(
                    pretrained_tokenizer.encode(review, max_length=512, truncation=True),
                    skip_special_tokens=True
                )
                pretrained_result = pretrained_analyzer(truncated_review)[0]
                label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

                results = {
                    "label": label_map.get(pretrained_result['label'], pretrained_result['label']),
                    "score": pretrained_result.get('score', 0),
                    "model_used": pretrained_model_path
                }

        response = JsonResponse(results)
        response["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        logger.error("Prediction error: %s", e)
        response = JsonResponse({"error": str(e)}, status=500)
        response["Access-Control-Allow-Origin"] = "*"
        return response

# Fetch Images API
@csrf_exempt
def get_all_images(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    try:
        sample_folder = os.path.join(settings.BASE_DIR, "sample-toys")
        images = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if not images:
            return JsonResponse({"error": "No images found."}, status=404)

        image_urls = [f"/sample-toys/{img}" for img in images]

        response = JsonResponse({"images": image_urls})
        response["Access-Control-Allow-Origin"] = "*"
        return response
    except Exception as e:
        logger.error("Error fetching all images: %s", e)
        response = JsonResponse({"error": str(e)}, status=500)
        response["Access-Control-Allow-Origin"] = "*"
        return response

# Clear Cache API
@csrf_exempt
def clear_cache(request):
    global pretrained_analyzer, pretrained_tokenizer, custom_model, custom_tokenizer, cached_custom_model_path
    pretrained_analyzer, pretrained_tokenizer = None, None
    custom_model, custom_tokenizer = None, None
    cached_custom_model_path = None
    logger.info("Model cache cleared.")
    return JsonResponse({"message": "Model cache cleared."})
