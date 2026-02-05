import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import time
import pandas as pd

# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="Social Bot Detection – FOX8",
    layout="wide"
)

st.title("🧠 Social Bot Detection – FOX8 Dataset")
st.caption(
    "Inference + Performance Evaluation on fox8-23 dataset "
    "(Fine-tuned Transformer Models)"
)

st.info("📊 Dataset: fox8-23 (Political & social media tweets)")

# ==================================================
# Text preprocessing (same as training)
# ==================================================
def preprocess_text(text):
    steps = []

    steps.append(("Original Text", text))

    text = re.sub(r"http\S+", "", text)
    steps.append(("URLs Removed", text))

    text = re.sub(r"<.*?>", "", text)
    steps.append(("HTML Removed", text))

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    steps.append(("Special Characters Removed", text))

    text = text.lower().strip()
    steps.append(("Lowercased Text", text))

    return steps, text

# ==================================================
# Model architecture (same as training)
# ==================================================
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

# ==================================================
# FOX8-trained models (UPDATE FILENAMES IF NEEDED)
# ==================================================
MODELS = {
    "BERT": ("bert-base-uncased", "bert_fox_model.pth"),
    "RoBERTa": ("roberta-base", "roberta_fox_model.pth"),
    "DistilBERT": ("distilbert-base-uncased", "distilbert_fox_model.pth"),
    "XLM-RoBERTa": ("xlm-roberta-base", "xlm_roberta_fox_model.pth")
}

# ==================================================
# Cache model loading
# ==================================================
@st.cache_resource
def load_model(model_name, weight_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BotDetectionModel(model_name)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model, tokenizer

# ==================================================
# Performance metrics (FOX8 – from paper)


# ==================================================
# User input
# ==================================================
st.subheader("🔤 Input Tweet")

tweet = st.text_area(
    "Enter a tweet to analyze",
    placeholder="The government policy will affect millions of citizens..."
)

# ==================================================
# Run pipeline (ALL MODELS)
# ==================================================
if st.button("🚀 Run All Models (FOX8)"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        progress = st.progress(0)

        # ---------------- Preprocessing ----------------
        st.subheader("🧹 Text Preprocessing")
        steps, cleaned_text = preprocess_text(tweet)

        for step, value in steps:
            st.markdown(f"**{step}:**")
            st.code(value)
            time.sleep(0.3)

        progress.progress(25)

        # ---------------- Inference ----------------
        st.subheader("🤖 Model Predictions (FOX8-trained)")
        results = []

        for model_label, (hf_name, weight_path) in MODELS.items():
            with st.spinner(f"Running {model_label}..."):
                model, tokenizer = load_model(hf_name, weight_path)

                inputs = tokenizer(
                    cleaned_text,
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

                confidence = float(output.item())
                prediction = "Bot 🤖" if confidence > 0.5 else "Human 👤"

                results.append({
                    "Model": model_label,
                    "Prediction": prediction,
                    "Confidence Score": round(confidence, 3)
                })

        progress.progress(60)

        st.table(pd.DataFrame(results))

       
