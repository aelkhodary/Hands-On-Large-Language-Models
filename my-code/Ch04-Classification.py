from datasets import load_dataset

# Load our data
data = load_dataset("rotten_tomatoes")
print(data["train"][0, -1])

from transformers import pipeline
import torch

# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Check if CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model into pipeline
try:
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device=device
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# Run inference
DEBUG_MODE = True  # Set to True to see detailed output for each prediction

y_pred = []
test_texts = data["test"]["text"]
for idx, output in enumerate(tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"]))):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    # Pick the higher score
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)
    
    # Optional debug output
    if DEBUG_MODE and idx < 5:  # Only show first 5 examples when debugging
        print(f"\nExample {idx + 1}:")
        print(f"Text: {test_texts[idx]}")
        print(f"Negative score: {negative_score:.4f}, Positive score: {positive_score:.4f}")
        print(f"Predicted: {assignment}, Actual: {data['test']['label'][idx]}")



from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)    

evaluate_performance(data["test"]["label"], y_pred)

