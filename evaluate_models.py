import torch
import torch.nn as nn
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cpu")

# --------------------------------------------------
# Load dataset (same preprocessing as training)
# --------------------------------------------------
df = pd.read_csv(
    "data/tweetfake.csv",
    sep=";",
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Select required columns
df = df[['text', 'account.type']]
df.columns = ['text', 'label']

# Convert labels
df['label'] = df['label'].astype(str).str.lower().map({
    'human': 0,
    'bot': 1
})

df = df.dropna()

# Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

df['text'] = df['text'].astype(str).apply(clean_text)

# Train-test split (same ratio)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# --------------------------------------------------
# Model architecture
# --------------------------------------------------
class BotDetectionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)

# --------------------------------------------------
# Models to evaluate
# --------------------------------------------------
MODELS = {
    "BERT": ("bert-base-uncased", "bert_bot_model.pth"),
    "RoBERTa": ("roberta-base", "roberta_bot_model.pth"),
    "DistilBERT": ("distilbert-base-uncased", "distilbert_bot_model.pth"),
    "XLM-RoBERTa": ("xlm-roberta-base", "xlm-roberta_bot_model.pth")
}

# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
for model_name, (hf_model, weight_file) in MODELS.items():
    print(f"\n🔹 Evaluating {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = BotDetectionModel(hf_model)
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    model.eval()

    y_true = []
    y_pred = []

    for text, label in zip(X_test, y_test):
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])

        prediction = 1 if output.item() > 0.5 else 0

        y_true.append(label)
        y_pred.append(prediction)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
