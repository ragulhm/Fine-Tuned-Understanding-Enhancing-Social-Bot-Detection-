# ============================================================
# BOT DETECTION — MULTI MODEL EVALUATION + ENSEMBLE (FIXED)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import numpy as np

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# LOAD TEST DATASET
# --------------------------------------------------
test_path = "/home/hitman/Documents/Capstone-Project-/Demo_Test/venv/bot_detection/retrained model/data/fox8_test.csv"
test_df = pd.read_csv(test_path)
test_df["label"] = test_df["label"].astype(int)

# --------------------------------------------------
# TEXT CLEANING (MATCH TRAINING)
# --------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return text.lower().strip()

test_df["tweet"] = test_df["tweet"].astype(str).apply(clean_text)

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# --------------------------------------------------
# MODEL (MATCHES SAVED WEIGHTS STRUCTURE)
# --------------------------------------------------
# --------------------------------------------------
# MODEL (MATCHES TRAINING ARCHITECTURE EXACTLY)
# --------------------------------------------------
class BotDetectionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        # EXACT MATCH with saved weights
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
MODELS = {
    "BERT_BASE": {
        "hf": "bert-base-uncased",
        "weights": "bert_fox_epoch.pth"
    },
    "DISTILBERT": {
        "hf": "distilbert-base-uncased",
        "weights": "distilbert_fox_model.pth"
    },
    "ROBERTA": {
        "hf": "roberta-base",
        "weights": "roberta_fox_model.pth"
    },
    "XLM_ROBERTA": {
        "hf": "xlm-roberta-base",
        "weights": "xlm_roberta_fox_model.pth"
    }
}

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
results = []
ensemble_probs = []
true_labels = None

for name, cfg in MODELS.items():

    if not os.path.exists(cfg["weights"]):
        print(f"❌ Skipping {name} — weights not found")
        continue

    print(f"\n🔍 Evaluating {name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf"])
    dataset = TweetDataset(test_df["tweet"], test_df["label"], tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = BotDetectionModel(cfg["hf"]).to(device)

    # LOAD WEIGHTS
    state = torch.load(cfg["weights"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].numpy()

            probs = model(ids, mask).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    results.append([name, acc, p, r, f1])
    ensemble_probs.append(all_probs)

    if true_labels is None:
        true_labels = all_labels

# --------------------------------------------------
# ENSEMBLE (AVERAGE)
# --------------------------------------------------
print("\n🧠 Running Ensemble Voting...")

ensemble_probs = np.mean(ensemble_probs, axis=0)
ensemble_preds = (ensemble_probs > 0.5).astype(int)

acc = accuracy_score(true_labels, ensemble_preds)
p, r, f1, _ = precision_recall_fscore_support(
    true_labels, ensemble_preds, average="binary"
)

results.append(["ENSEMBLE", acc, p, r, f1])

# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
)

print("\n===== FINAL COMPARISON =====")
print(results_df)

results_df.to_csv("ensemble_results.csv", index=False)
print("\n✅ Results saved → ensemble_results.csv")
