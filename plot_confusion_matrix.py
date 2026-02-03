import torch
import torch.nn as nn
import pandas as pd
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# Device (CPU is enough)
# --------------------------------------------------
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

df.columns = [c.strip().lower() for c in df.columns]

df = df[['text', 'account.type']]
df.columns = ['text', 'label']

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

# Train-test split
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
# Models to plot
# --------------------------------------------------
MODELS = {
    "BERT": ("bert-base-uncased", "bert_bot_model.pth"),
    "RoBERTa": ("roberta-base", "roberta_bot_model.pth"),
    "DistilBERT": ("distilbert-base-uncased", "distilbert_bot_model.pth"),
    "XLM-RoBERTa": ("xlm-roberta-base", "xlm-roberta_bot_model.pth")
}

# --------------------------------------------------
# Plot confusion matrix function
# --------------------------------------------------
def plot_cm(cm, model_name):
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Human", "Bot"])
    plt.yticks([0, 1], ["Human", "Bot"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()

# --------------------------------------------------
# Generate confusion matrix for each model
# --------------------------------------------------
for model_name, (hf_model, weight_file) in MODELS.items():
    print(f"\nGenerating Confusion Matrix for {model_name}")

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

    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, model_name)

print("\n✅ Confusion matrix plots generated and saved.")
 
