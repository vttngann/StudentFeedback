import torch

MODEL_PATH = "phobert_multitask_best.pt"

TOKENIZER_PATH = "phobert_tokenizer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 256

SENTIMENT_LABELS = {
    0: "Tiêu cực",
    1: "Trung lập",
    2: "Tích cực"
}

TOPIC_LABELS = {
    0: "Giảng viên",
    1: "Chương trình học",
    2: "Cơ sở vật chất",
    3: "Khác"
}