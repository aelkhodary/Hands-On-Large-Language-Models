# !pip install -q transformers datasets scikit-learn accelerate

import random, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, pipeline)
import torch

# ---------------------------
# 0) Reproducibility & Device
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = 0 if torch.cuda.is_available() else -1

# -----------------------------------
# 1) Generate a small synthetic dataframe
# -----------------------------------
# Synonyms / patterns per city (add/adjust to match your real data)
dubai_texts = [
    "Shipment to Dubai", "Visited DXB airport", "Delivery in dub", "Burj Khalifa Dubai",
    "Address: Deira, Dubai", "Dubai Marina order", "from dxb", "City=DUBAI"
]
abudhabi_texts = [
    "Delivery to Abu Dhabi", "Address: Al Reem Island Abu Dhabi", "Shipment AUH",
    "from abudhabi", "Abu Dhabi Corniche", "City=Abu Dhabi", "AUH office"
]
cairo_texts = [
    "Shipment to Cairo", "Address: Maadi, Cairo", "from cai", "Cairo Egypt order",
    "Nasr City Cairo", "City=CAIRO", "Delivery in Giza Cairo"
]
london_texts = [
    "Delivery to London", "Address: Kensington London", "from lon", "Oxford Street London",
    "City=LONDON", "Shipment to LDN", "London UK order"
]

def synthesize(city, candidates, n=80):
    rows = []
    for _ in range(n):
        t = random.choice(candidates)
        # simple noise/variants
        if random.random() < 0.3:
            t = t.lower()
        if random.random() < 0.2:
            t = f"{t} #{random.randint(1,9999)}"
        rows.append({"text": t, "label": city})
    return rows

rows = []
rows += synthesize("Dubai", dubai_texts, n=80)
rows += synthesize("Abu Dhabi", abudhabi_texts, n=80)
rows += synthesize("Cairo", cairo_texts, n=80)
rows += synthesize("London", london_texts, n=80)

df = pd.DataFrame(rows)
print("Sample df:\n", df.head())

# -----------------------------------
# 2) Train/Val/Test split (stratified)
# -----------------------------------
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["label"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

# -----------------------------------
# 3) Build HF Datasets and encode labels
# -----------------------------------
ds = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
})

# Turn string labels into integer ids consistently
ds = ds.class_encode_column("label")
labels = ds["train"].features["label"].names
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}
print("Label mapping:", label2id)

# -----------------------------------
# 4) Tokenizer & Model
# -----------------------------------
MODEL_NAME = "distilbert-base-uncased"
# For Arabic/multilingual, use: MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tok, batched=True)
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("input_ids","attention_mask","label")])
ds.set_format("torch")

# -----------------------------------
# 5) Metrics
# -----------------------------------
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# -----------------------------------
# 6) Training
# -----------------------------------
args = TrainingArguments(

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="no",
    load_best_model_at_end=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------------
# 7) Evaluation on TEST
# -----------------------------------
test_preds = trainer.predict(ds["test"])
test_y_true = test_preds.label_ids
test_y_pred = test_preds.predictions.argmax(axis=-1)
print("\n=== Classification Report (TEST) ===")
print(classification_report(test_y_true, test_y_pred, target_names=labels, digits=3))

# -----------------------------------
# 8) Inference: try new strings
# -----------------------------------
clf = pipeline("text-classification", model=trainer.model, tokenizer=tokenizer, device=device, return_all_scores=True)

examples = [
    "al reem island Abu Dhabi",
    "Oxford Street London",
    "from dxb",
    "Maadi Cairo",
    "burj khalifa dubai",
    "city=ldn"
]

print("\n=== Inference Examples ===")
for x in examples:
    out = clf(x)
    # find top label & score
    best = max(out[0], key=lambda d: d["score"])
    print(f"{x!r} -> {best['label']} (p={best['score']:.3f})")


# -----------------------------------
# 9) Save for later use (optional)
# -----------------------------------
save_dir = "/content/city-clf"
os.makedirs(save_dir, exist_ok=True)
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)



# Reload example:
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
tok = AutoTokenizer.from_pretrained("/content/city-clf")
mdl = AutoModelForSequenceClassification.from_pretrained("city-clf")
clf = pipeline("text-classification", model=mdl, tokenizer=tok, device=device, return_all_scores=True)



text = "cairo"
out = clf(text)[0]                       # list of {label, score} dicts
best = max(out, key=lambda d: d["score"])
print(best["label"], round(best["score"], 3))
# e.g., -> "Dubai" 0.994
