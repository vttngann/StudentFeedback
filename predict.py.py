import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import PhoBERTMultiTask
from config import *

# ==========================
# Load Tokenizer
# ==========================

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# ==========================
# Load Model
# ==========================

model = PhoBERTMultiTask()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


# ==========================
# Predict
# ==========================

def predict(text):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():

        sentiment_logits, topic_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    sentiment_prob = F.softmax(sentiment_logits, dim=1)
    topic_prob = F.softmax(topic_logits, dim=1)

    sentiment_id = sentiment_prob.argmax().item()
    topic_id = topic_prob.argmax().item()

    return {

        "sentiment": SENTIMENT_LABELS[sentiment_id],

        "topic": TOPIC_LABELS[topic_id],

        # app.py đang dùng các key này
        "sentiment_confidence":
            round(sentiment_prob.max().item() * 100, 2),

        "topic_confidence":
            round(topic_prob.max().item() * 100, 2),

        # vì model chưa dự đoán emotion
        "emotion": "Không xác định"
    }