from datasets import load_dataset


from transformers import pipeline
import torch
from sklearn.metrics import classification_report

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# **Text Classification with Representation Models**
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch



class TextClassificationWithRepresentationModels:
    def __init__(self, model_path="cardiffnlp/twitter-roberta-base-sentiment-latest", debug_mode=True):
        self.model_path = model_path
        self.debug_mode = debug_mode
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        try:
            self.pipe = pipeline(
                model=self.model_path,
                tokenizer=self.model_path,
                return_all_scores=True,
                device=self.device
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, dataset, text_column="text", label_column="label", max_debug_examples=5):
        y_pred = []
        test_texts = dataset[text_column]
        for idx, output in enumerate(tqdm(self.pipe(KeyDataset(dataset, text_column)), total=len(dataset))):
            negative_score = output[0]["score"]
            positive_score = output[2]["score"]
            # Pick the higher score
            assignment = np.argmax([negative_score, positive_score])
            y_pred.append(assignment)
            
            # Optional debug output
            if self.debug_mode and idx < max_debug_examples:
                print(f"\nExample {idx + 1}:")
                print(f"Text: {test_texts[idx]}")
                print(f"Negative score: {negative_score:.4f}, Positive score: {positive_score:.4f}")
                print(f"Predicted: {assignment}, Actual: {dataset[label_column][idx]}")
        return y_pred



class SupervisedClassification:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', random_state=42, debug_mode=True):
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.clf = LogisticRegression(random_state=random_state)
        self.debug_mode = debug_mode

    def fit(self, train_texts, train_labels):
        # Convert training text to embeddings
        train_embeddings = self.model.encode(train_texts, show_progress_bar=True)
        # Train the classifier
        self.clf.fit(train_embeddings, train_labels)

    def predict(self, dataset, text_column="text", label_column="label", max_debug_examples=5):
        # Convert test text to embeddings
        test_embeddings = self.model.encode(dataset[text_column], show_progress_bar=True)
        # Predict labels
        y_pred = self.clf.predict(test_embeddings)
        
        if self.debug_mode:
            for idx in range(min(max_debug_examples, len(dataset[text_column]))):
                print(f"\nExample {idx + 1}:")
                print(f"Text: {dataset[text_column][idx]}")
                print(f"Predicted: {y_pred[idx]}")
                print(f"Actual: {dataset[label_column][idx]}", end="\n")
                
        return y_pred

       



def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)    



def main():
    # Load our data
    data = load_dataset("rotten_tomatoes")
    print(data["train"][0, -1])

    # Example usage for TextClassificationWithRepresentationModels (if implemented elsewhere)
    # classifier = TextClassificationWithRepresentationModels()
    # y_pred = classifier.predict(data["test"])
    # evaluate_performance(data["test"]["label"], y_pred)

    # Example usage for SupervisedClassification
    sc = SupervisedClassification()
    sc.fit(data["train"]["text"], data["train"]["label"])
    y_pred = sc.predict(data["test"])
    evaluate_performance(data["test"]["label"], y_pred)

if __name__ == "__main__":
    main()
