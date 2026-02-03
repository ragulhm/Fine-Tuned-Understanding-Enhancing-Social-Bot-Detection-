import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import re

from tqdm import tqdm


# --------------------------------------------------
# Device configuration
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Load dataset (TSV / messy Twitter-safe)
# --------------------------------------------------
df = pd.read_csv(
    "data/tweetfake.csv",
    sep=";",              # <-- CRITICAL FIX
    header=0,             # <-- use first row as header
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

print("Columns:", df.columns)
print(df.head(3))




# --------------------------------------------------
# Normalize column names
# --------------------------------------------------
df.columns = [c.strip().lower() for c in df.columns]
print("Detected columns:", df.columns.tolist())

# --------------------------------------------------
# Auto-detect text and label columns
# --------------------------------------------------
text_col = None
label_col = None

for c in df.columns:
    if c in ["text", "tweet", "tweet_text", "content"]:
        text_col = c
        break

for c in df.columns:
    if c in ["account.type", "account_type", "type", "label", "class"]:
        label_col = c
        break

if text_col is None or label_col is None:
    raise ValueError("❌ Could not detect text or label column")

# keep only required columns
df = df[['text', 'account.type']]

# rename for model
df.columns = ['text', 'label']

# convert labels to numeric
df['label'] = df['label'].map({
    'human': 0,
    'bot': 1
})

# drop bad rows
df = df.dropna()

# --------------------------------------------------
# Text cleaning (recommended)
# --------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

df["text"] = df["text"].astype(str).apply(clean_text)

print(df.head())
print("Label distribution:\n", df["label"].value_counts())

# --------------------------------------------------
# Train–test split (80–20)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)
print(df["X_test"]).head()
# --------------------------------------------------
# Transformer models
# # --------------------------------------------------
# MODEL_PATHS = {
#     "bert": "bert-base-uncased",
#     "roberta": "roberta-base",
#     "distilbert": "distilbert-base-uncased",
#     "xlm-roberta": "xlm-roberta-base"
# }

# # --------------------------------------------------
# # Dataset class
# # --------------------------------------------------
# class TweetDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             self.texts.iloc[idx],
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_len,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "label": torch.tensor(self.labels.iloc[idx], dtype=torch.float)
#         }

# # --------------------------------------------------
# # Model architecture
# # --------------------------------------------------
# class BotDetectionModel(nn.Module):
#     def __init__(self, model_name):
#         super().__init__()

#         self.encoder = AutoModel.from_pretrained(model_name)
#         hidden_size = self.encoder.config.hidden_size

#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, input_ids, attention_mask):
#         outputs = self.encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
#         return self.classifier(cls_embedding)

# # --------------------------------------------------
# # Training function
# # --------------------------------------------------
# def train_model(model, dataloader, optimizer, criterion, epochs=3):
#     model.train()

#     for epoch in range(epochs):
#         total_loss = 0

#         for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True):
#           input_ids = batch["input_ids"].to(device)
#           attention_mask = batch["attention_mask"].to(device)
#           labels = batch["label"].to(device).unsqueeze(1)

#           optimizer.zero_grad()
#           outputs = model(input_ids, attention_mask)
#           loss = criterion(outputs, labels)

#           loss.backward()
#           optimizer.step()

#           total_loss += loss.item()
  

# # --------------------------------------------------
# # Train all models
# # --------------------------------------------------
# for model_key, model_path in MODEL_PATHS.items():
#     print(f"\nTraining {model_key.upper()} model")

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     train_dataset = TweetDataset(X_train, y_train, tokenizer)
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#     model = BotDetectionModel(model_path).to(device)
#     optimizer = AdamW(model.parameters(), lr=2e-5)
#     criterion = nn.BCELoss()

#     train_model(model, train_loader, optimizer, criterion)
#     torch.save(model.state_dict(), f"{model_key}_bot_model.pth")

# # --------------------------------------------------
# # Prediction function
# # --------------------------------------------------
# def predict_tweet(model_path, model_weights, tweet):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = BotDetectionModel(model_path)
#     model.load_state_dict(torch.load(model_weights, map_location=device))
#     model.to(device)
#     model.eval()

#     encoding = tokenizer(
#         tweet,
#         truncation=True,
#         padding="max_length",
#         max_length=128,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         output = model(
#             encoding["input_ids"].to(device),
#             encoding["attention_mask"].to(device)
#         )

#     return "Bot" if output.item() > 0.5 else "Human"
