import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

device = torch.device("cpu")

# -------------------------------
# Model architecture
# -------------------------------
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

# -------------------------------
# All trained models
# -------------------------------
MODELS = {
    "BERT": ("bert-base-uncased", "bert_bot_model.pth"),
    "RoBERTa": ("roberta-base", "roberta_bot_model.pth"),
    "DistilBERT": ("distilbert-base-uncased", "distilbert_bot_model.pth"),
    "XLM-RoBERTa": ("xlm-roberta-base", "xlm-roberta_bot_model.pth")
}

# -------------------------------
# Prediction using all models
# -------------------------------
def predict_all(tweet):
    results = {}

    for name, (hf_model, weight_file) in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = BotDetectionModel(hf_model)
        model.load_state_dict(torch.load(weight_file, map_location="cpu"))
        model.eval()

        inputs = tokenizer(
            tweet,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])

        results[name] = "Bot 🤖" if output.item() > 0.5 else "Human 👤"

    return results

# -------------------------------
# User input
# -------------------------------
if __name__ == "__main__":
    tweet = input("Enter tweet: ")
    predictions = predict_all(tweet)

    print("\n--- MODEL PREDICTIONS ---")
    for model, result in predictions.items():
        print(f"{model}: {result}")
