import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Load TEST dataset (20%)
# --------------------------------------------------
test_df = pd.read_csv("/home/hitman/Documents/Capstone-Project-/Demo_Test/venv/bot_detection/retrained model/data/tweetfake_test.csv")
test_df["label"] = test_df["label"].astype(int)

# --------------------------------------------------
# Text cleaning (MUST MATCH TRAINING)
# --------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

test_df["tweet"] = test_df["tweet"].astype(str).apply(clean_text)

# --------------------------------------------------
# Dataset class
# --------------------------------------------------
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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
# Model architecture (SAME AS TRAINING)
# --------------------------------------------------
class BotDetectionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# --------------------------------------------------
# ALL MODELS TO EVALUATE
# --------------------------------------------------
MODELS = {
    "BERT": {
        "hf": "bert-base-uncased",
        "weights": "/home/hitman/Documents/Capstone-Project-/Demo_Test/venv/bot_detection/retrained model/models/bert_tweetfake_model.pth"
    },
    
    "DistilBERT": {
        "hf": "distilbert-base-uncased",
        "weights": "/home/hitman/Documents/Capstone-Project-/Demo_Test/venv/bot_detection/retrained model/models/distilbert_tweetfake_model.pth"
    },
    "RoBERTa": {
        "hf": "roberta-base",
        "weights": "/home/hitman/Documents/Capstone-Project-/Demo_Test/venv/bot_detection/retrained model/models/roberta_tweetfake_model.pth"
    },
    "XLM-RoBERTa": {
        "hf": "xlm-roberta-base",
        "weights": "/home/hitman/Documents/Capstone-Project-/Demo_Test/venv/bot_detection/retrained model/models/xlm-roberta_tweetfake_model.pth"
    }
}

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
results = []

for name, cfg in MODELS.items():
    if not os.path.exists(cfg["weights"]):
        print(f"❌ Skipping {name} (model not found)")
        continue

    print(f"\n🔍 Evaluating {name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf"])
    dataset = TweetDataset(test_df["tweet"], test_df["label"], tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = BotDetectionModel(cfg["hf"]).to(device)
    model.load_state_dict(torch.load(cfg["weights"], map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(ids, mask)
            preds = (outputs > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    results.append([name, acc, p, r, f1])

# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
)

print("\n===== FINAL COMPARISON =====")
print(results_df)

results_df
