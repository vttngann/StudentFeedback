import torch
import torch.nn.functional as F
from transformers import PhobertTokenizer

from model import PhoBERTMultiTask
from config import *

# Device
device = torch.device(DEVICE)

# Load tokenizer
tokenizer = PhobertTokenizer.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True
)
# Load model
model = PhoBERTMultiTask()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()


def predict(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():

        sentiment_logits, topic_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sentiment_prob = F.softmax(sentiment_logits, dim=1)
        topic_prob = F.softmax(topic_logits, dim=1)

    sentiment_id = sentiment_prob.argmax(dim=1).item()
    topic_id = topic_prob.argmax(dim=1).item()

    return {

        "sentiment": SENTIMENT_LABELS[sentiment_id],
        "topic": TOPIC_LABELS[topic_id],

        "sentiment_confidence":
            round(sentiment_prob.max().item()*100,2),

        "topic_confidence":
            round(topic_prob.max().item()*100,2)

    }