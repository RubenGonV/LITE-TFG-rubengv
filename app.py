import logging
import os
import sys
import time
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load CLIP model once at the start
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_imagenet_labels():
    try:
        with open('imagenet_labels.txt', 'r') as f:
            labels = f.read().splitlines()
        return labels
    except Exception as e:
        logging.error("Error loading ImageNet labels: %s", e)
        return []

def clip_classify(image_array, text_labels):
    image = Image.fromarray(image_array.astype('uint8'))
    inputs = clip_processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    start_time = time.time()
    outputs = clip_model(**inputs)
    end_time = time.time()
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()
    elapsed_time = end_time - start_time
    return probs, elapsed_time

def get_top_predictions(predictions, top_k=3):
    class_indices = np.argsort(predictions[0])[::-1][:top_k]
    return [(i, predictions[0][i]) for i in class_indices]

@app.route('/')
def index():
    return render_template('select_dataset.html')

@app.route('/classify_custom', methods=['GET', 'POST'])
def classify_custom():
    if request.method == 'POST':
        logging.debug("Received a POST request for custom label classification")
        file = request.files.get('file_custom')
        custom_labels_only = request.form.get('custom_labels_only')

        if file:
            if custom_labels_only:
                labels = [label.strip() for label in custom_labels_only.split(',')]
                logging.debug("Using custom labels for classification: %s", labels)
            else:
                labels = load_imagenet_labels()
                logging.debug("Using ImageNet labels for classification")

            try:
                image = Image.open(file.stream)
                image = image.convert('RGB')
                image_array = np.array(image)

                clip_probs, clip_time = clip_classify(image_array, labels)
                clip_top_preds = get_top_predictions(np.array([clip_probs[0]]))
                clip_top_preds = [(labels[idx], prob) for idx, prob in clip_top_preds]

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                result = {
                    'image_data': img_str,
                    'clip_predictions': clip_top_preds,
                    'clip_time': clip_time,
                }

                return render_template('classify_custom.html', result=result)
            except Exception as e:
                logging.error("Error processing uploaded image with custom labels: %s", e)
                return render_template('select_dataset.html', error="Error processing uploaded image with custom labels.")

    return render_template('select_dataset.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

