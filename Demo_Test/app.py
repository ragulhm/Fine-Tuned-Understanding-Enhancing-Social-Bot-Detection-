import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

device = torch.device("cpu")

# --------------------------------------------------
# Model architecture (same as training)
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
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)

# --------------------------------------------------
# Model configuration (ALL 4 MODELS)
# --------------------------------------------------
MODELS = {
    "BERT": {
        "name": "bert-base-uncased",
        "weights": "bert_bot_model.pth"
    },
    "RoBERTa": {
        "name": "roberta-base",
        "weights": "roberta_bot_model.pth"
    },
    "DistilBERT": {
        "name": "distilbert-base-uncased",
        "weights": "distilbert_bot_model.pth"
    },
    "XLM-RoBERTa": {
        "name": "xlm-roberta-base",
        "weights": "xlm-roberta_bot_model.pth"
    }
}

# --------------------------------------------------
# Cache model loading (VERY IMPORTANT)
# --------------------------------------------------
@st.cache_resource
def load_model(model_name, weight_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BotDetectionModel(model_name)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model, tokenizer

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Social Bot Detection", layout="centered")

st.title("🕵️ Social Bot Detection System")
st.write("Predict whether a tweet is **Human or Bot** using transformer models.")

tweet = st.text_area("Enter Tweet Text")

model_choice = st.selectbox(
    "Select Model",
    list(MODELS.keys())
)

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        model_info = MODELS[model_choice]
        model, tokenizer = load_model(
            model_info["name"],
            model_info["weights"]
        )

        inputs = tokenizer(
            tweet,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )

        result = "Bot 🤖" if output.item() > 0.5 else "Human 👤"

        st.success(f"Model Used: {model_choice}")
        st.success(f"Prediction: {result}")
