import asyncio
import random
import os
import json
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import các thư viện xử lý KeyBERT và KMeans
from keybert import KeyBERT
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from database import (
    init_db,
    save_feedback,
    get_feedback,
    delete_feedback,
    update_status,
    get_statistics
)

from predict import predict

# Khai báo mô hình KeyBERT ban đầu dưới dạng None (Khởi tạo ở Lifespan)
kw_model = None

# Cache biến toàn cục tối ưu hiệu năng
CACHE_KEYWORDS = {"data": [], "comment_count": -1}
CACHE_CLUSTERS = {"data": [], "timestamp": 0, "comment_count": -1}


def invalidate_caches():
    """Reset lại cache khi có sự thay đổi dữ liệu (thêm/sửa/xóa comment)"""
    CACHE_KEYWORDS["comment_count"] = -1
    CACHE_CLUSTERS["comment_count"] = -1


# =====================================================
# PYDANTIC SCHEMAS
# =====================================================

class PredictRequest(BaseModel):
    text: str
    faculty: str = "CNTT & TT"


# =====================================================
# TẠO THƯ MỤC NỀN TẢNG
# =====================================================

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def parse_sentiment(sentiment_raw: str) -> str:
    """Chuẩn hóa so sánh Sentiment an toàn"""
    s = str(sentiment_raw).strip().lower()
    if any(k in s for k in ["tích cực", "positive", "pos", "tich cuc"]):
        return "Tích cực"
    elif any(k in s for k in ["tiêu cực", "negative", "neg", "tieu cuc"]):
        return "Tiêu cực"
    return "Trung tính"


def calculate_priority(sentiment: str) -> str:
    """Tự động tính toán mức độ ưu tiên theo sắc thái"""
    norm = parse_sentiment(sentiment)
    if norm == "Tiêu cực":
        return "HIGH"
    elif norm == "Trung tính":
        return "MEDIUM"
    return "LOW"


def load_model_metrics():
    """Đọc trực tiếp từ file kết quả đánh giá thực tế của mô hình PhoBERT."""
    metrics_path = 'model_metrics.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Lỗi đọc file model_metrics.json: {e}")
            
    return {
        "sentiment": {"precision": "N/A", "recall": "N/A", "f1": "N/A"},
        "topic": {"precision": "N/A", "recall": "N/A", "f1": "N/A"},
        "avg_f1": "N/A",
        "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    }


# =====================================================
# WEBSOCKET MANAGER
# =====================================================

class ConnectionManager:

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


# =====================================================
# BOT GIẢ LẬP REALTIME
# =====================================================

MOCK_COMMENTS = [
    "Wifi thư viện rất yếu.",
    "Máy chiếu bị hỏng.",
    "Thầy dạy rất hay.",
    "Điều hòa phòng học không mát.",
    "Đồ ăn căn tin ngon.",
    "Cần thêm chỗ gửi xe."
]

FACULTIES = ["CNTT & TT", "Kinh tế", "Ngoại ngữ", "Luật", "Kỹ thuật - Công nghệ"]


async def simulated_student_stream():
    while True:
        await asyncio.sleep(random.randint(4, 7))

        if manager.active_connections:
            text = random.choice(MOCK_COMMENTS)
            faculty = random.choice(FACULTIES)

            result = predict(text)
            result["emotion"] = "Không xác định"
            result["faculty"] = faculty

            save_feedback(text, result)
            invalidate_caches()

            sentiment = result.get("sentiment", "Trung tính")
            priority = result.get("priority") or calculate_priority(sentiment)

            payload = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "user": f"SV_{random.randint(1000, 9999)}",
                "faculty": faculty,
                "text": text,
                "topic": result.get("topic", "Khác"),
                "priority": priority,
                "confidence": result.get("sentiment_confidence", 90),
                "sentiment": parse_sentiment(sentiment),
                "emotion": result.get("emotion", "Không xác định")
            }

            await manager.broadcast(payload)


# =====================================================
# LIFESPAN MANAGEMENT
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kw_model
    init_db()
    
    # Khởi tạo mô hình KeyBERT tại lifespan để tránh nghẽn khi import app
    print("Loading KeyBERT Model...")
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    print("KeyBERT Model Loaded successfully.")

    task = asyncio.create_task(simulated_student_stream())
    yield
    task.cancel()


# =====================================================
# FASTAPI APP SETUP
# =====================================================

app = FastAPI(lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")


# =====================================================
# WEBSOCKET ENDPOINT
# =====================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Lắng nghe tin nhắn/ping thực tế từ client
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# =====================================================
# GIAO DIỆN (HTML ROUTES)
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/student", response_class=HTMLResponse)
async def student(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/comments", response_class=HTMLResponse)
async def comments(request: Request):
    return templates.TemplateResponse("comments.html", {"request": request})


@app.get("/statistics", response_class=HTMLResponse)
async def statistics(request: Request):
    feedbacks = get_feedback()
    stats_dict = {}
    
    total_feedback = len(feedbacks)
    total_pos = 0
    total_neg = 0
    
    for fb in feedbacks:
        group_name = fb.get("topic", "Khác")
        
        if group_name not in stats_dict:
            stats_dict[group_name] = {
                "total": 0, "positive": 0, "negative": 0, "neutral": 0, "resolved": 0
            }
            
        stats_dict[group_name]["total"] += 1
        sentiment_norm = parse_sentiment(fb.get("sentiment", ""))
        
        if sentiment_norm == "Tích cực":
            stats_dict[group_name]["positive"] += 1
            total_pos += 1
        elif sentiment_norm == "Tiêu cực":
            stats_dict[group_name]["negative"] += 1
            total_neg += 1
        else:
            stats_dict[group_name]["neutral"] += 1
            
        status = fb.get("status", "")
        if status in ["Đã xử lý", "Hoàn thành"]:
            stats_dict[group_name]["resolved"] += 1

    faculty_stats = []
    color_palette = ["bg-blue-400", "bg-purple-400", "bg-amber-400", "bg-cyan-400", "bg-rose-400"]
    
    for idx, (name, data) in enumerate(stats_dict.items()):
        total = data["total"]
        if total == 0: 
            continue
        
        pos_percent = round((data["positive"] / total) * 100, 1)
        neg_percent = round((data["negative"] / total) * 100, 1)
        res_rate = round((data["resolved"] / total) * 100, 1)
        
        if pos_percent > 60:
            eval_text, status_color = "Tốt", "emerald"
        elif neg_percent > 30:
            eval_text, status_color = "Cần lưu ý", "red"
        else:
            eval_text, status_color = "Khá", "amber"
            
        faculty_stats.append({
            "name": name,
            "dot_color": color_palette[idx % len(color_palette)],
            "total": total,
            "positive": data["positive"],
            "positive_percent": pos_percent,
            "negative": data["negative"],
            "negative_percent": neg_percent,
            "neutral": data["neutral"],
            "resolution_rate": res_rate,
            "status_color": status_color,
            "evaluation": eval_text
        })
        
    faculty_stats.sort(key=lambda x: x["total"], reverse=True)

    positive_rate = f"{round((total_pos / total_feedback)*100, 1)}%" if total_feedback > 0 else "0%"
    negative_rate = f"{round((total_neg / total_feedback)*100, 1)}%" if total_feedback > 0 else "0%"

    metrics = load_model_metrics()

    return templates.TemplateResponse(
        "statistics.html", 
        {
            "request": request, 
            "faculty_stats": faculty_stats,
            "metrics": metrics,
            "total_feedback": total_feedback,
            "positive_rate": positive_rate,
            "negative_rate": negative_rate
        }
    )


@app.get("/wordcloud", response_class=HTMLResponse)
async def wordcloud(request: Request):
    return templates.TemplateResponse("wordcloud.html", {"request": request})


@app.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    return templates.TemplateResponse("reports.html", {"request": request})


# =====================================================
# API ENDPOINTS
# =====================================================

@app.post("/api/upload-metrics")
async def upload_metrics(file: UploadFile = File(...)):
    if not file.filename.endswith('.json'):
        return JSONResponse(
            status_code=400, 
            content={"success": False, "message": "Chỉ chấp nhận file định dạng .json!"}
        )
    
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        # Validation cấu trúc file JSON
        required_keys = ["sentiment", "topic", "avg_f1", "matrix"]
        for key in required_keys:
            if key not in data:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"File JSON thiếu trường cấu trúc: '{key}'"}
                )

        with open('model_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        return {"success": True, "message": "Cập nhật chỉ số mô hình thành công!"}
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"success": False, "message": f"File JSON không hợp lệ: {str(e)}"}
        )


@app.post("/predict")
async def classify(data: PredictRequest):
    # Validation nội dung nhập
    if not data.text.strip():
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Nội dung phản hồi không được để trống."
            }
        )

    text = data.text
    faculty = data.faculty

    result = predict(text)
    result["emotion"] = "Không xác định"
    result["faculty"] = faculty

    sentiment = result.get("sentiment", "Trung tính")
    priority = result.get("priority") or calculate_priority(sentiment)
    result["priority"] = priority

    save_feedback(text, result)
    invalidate_caches()

    payload = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "user": "Sinh viên",
        "faculty": faculty,
        "text": text,
        "topic": result.get("topic", "Khác"),
        "priority": priority,
        "confidence": result.get("sentiment_confidence", 90),
        "sentiment": parse_sentiment(sentiment),
        "emotion": result.get("emotion", "Không xác định")
    }

    await manager.broadcast(payload)

    return result


@app.get("/api/dashboard")
async def get_dashboard_summary():
    feedbacks = get_feedback()

    total = len(feedbacks)
    positive = 0
    negative = 0
    neutral = 0

    topic_counts = {}
    top_negative = {}

    for fb in feedbacks:
        sentiment = parse_sentiment(fb.get("sentiment", ""))
        topic = fb.get("topic", "Khác")

        topic_counts[topic] = topic_counts.get(topic, 0) + 1

        if sentiment == "Tích cực":
            positive += 1
        elif sentiment == "Tiêu cực":
            negative += 1
            top_negative[topic] = top_negative.get(topic, 0) + 1
        else:
            neutral += 1

    total_calc = max(total, 1)

    top_negative_topics = []
    for k, v in top_negative.items():
        top_negative_topics.append({
            "topic": k,
            "percentage": round(v / negative * 100, 1) if negative else 0
        })

    # Sắp xếp các chủ đề tiêu cực giảm dần theo tỷ lệ phần trăm
    top_negative_topics.sort(key=lambda x: x["percentage"], reverse=True)

    return {
        "total": positive + negative + neutral,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_rate": positive * 100 / total_calc,
        "negative_rate": negative * 100 / total_calc,
        "neutral_rate": neutral * 100 / total_calc,
        "topic_counts": topic_counts,
        "topics": topic_counts,
        "top_negative_topics": top_negative_topics
    }


@app.get("/api/comments")
async def get_comments_list(limit: int = 10):
    feedbacks = get_feedback()
    
    # Sắp xếp chuẩn theo mốc thời gian created_at mới nhất
    recent_feedbacks = sorted(
        feedbacks,
        key=lambda x: str(x.get("created_at", "")),
        reverse=True
    )[:limit]

    result = []
    for fb in recent_feedbacks:
        sentiment = parse_sentiment(fb.get("sentiment", "Trung tính"))
        priority = fb.get("priority") or calculate_priority(sentiment)
        
        result.append({
            "time": fb.get("created_at", datetime.now().strftime("%H:%M:%S")),
            "user": fb.get("user_id", "SV"),
            "faculty": fb.get("faculty", "CNTT & TT"),
            "text": fb.get("comment", ""),
            "topic": fb.get("topic", "Khác"),
            "priority": priority,
            "confidence": fb.get("sentiment_confidence", 90),
            "sentiment": sentiment,
            "conf": fb.get("sentiment_confidence", 90),
            "emotion": fb.get("emotion", "Không xác định")
        })
    return result


@app.get("/api/keywords")
async def get_keywords():
    global CACHE_KEYWORDS, kw_model
    feedbacks = get_feedback()
    if not feedbacks:
        return []

    # Sử dụng Cache nếu số lượng comment chưa thay đổi
    current_count = len(feedbacks)
    if CACHE_KEYWORDS["comment_count"] == current_count:
        return CACHE_KEYWORDS["data"]

    full_text = " ".join([fb.get("comment", "") for fb in feedbacks if fb.get("comment")])
    if not full_text.strip():
        return []

    if kw_model is None:
        kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

    keywords_extracted = kw_model.extract_keywords(
        full_text,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=12
    )

    total = sum(score for _, score in keywords_extracted) if keywords_extracted else 1
    res = [
        {
            "word": word,
            "keyword": word,
            "count": int(score * 100),
            "percent": round(score / total * 100, 1)
        }
        for word, score in keywords_extracted
    ]

    CACHE_KEYWORDS["data"] = res
    CACHE_KEYWORDS["comment_count"] = current_count

    return res


@app.get("/api/issues")
@app.get("/api/clusters")
async def get_issues():
    global CACHE_CLUSTERS
    feedbacks = get_feedback()
    current_count = len(feedbacks)
    now = time.time()

    # Cache 5 phút (300s) hoặc khi số lượng comment không đổi
    if (now - CACHE_CLUSTERS["timestamp"] < 300) and (CACHE_CLUSTERS["comment_count"] == current_count):
        return CACHE_CLUSTERS["data"]

    negative_comments = [
        fb for fb in feedbacks 
        if parse_sentiment(fb.get("sentiment", "")) == "Tiêu cực"
    ]

    if len(negative_comments) < 2:
        return []

    texts = [fb.get("comment", "") for fb in negative_comments]
    
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    num_clusters = min(3, len(texts))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    clusters_result = []
    for i in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue

        centroid = kmeans.cluster_centers_[i]
        top_keyword_indices = centroid.argsort()[-3:][::-1]
        keywords = [feature_names[idx] for idx in top_keyword_indices if idx < len(feature_names)]

        topics = [negative_comments[idx].get("topic", "Khác") for idx in cluster_indices]
        main_topic = max(set(topics), key=topics.count) if topics else "Khác"

        clusters_result.append({
            "topic": main_topic,
            "count": len(cluster_indices),
            "keywords": keywords,
            "negative_percent": 100
        })

    CACHE_CLUSTERS["data"] = clusters_result
    CACHE_CLUSTERS["timestamp"] = now
    CACHE_CLUSTERS["comment_count"] = current_count

    return clusters_result


@app.get("/api/heatmap")
async def get_heatmap():
    """Thống kê Heatmap THẬT từ Database theo Thứ và Khung giờ"""
    feedbacks = get_feedback()
    matrix = [[0] * 7 for _ in range(3)]
    
    for fb in feedbacks:
        created_str = fb.get("created_at", "")
        try:
            dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            # Bỏ qua record nếu sai format thời gian thay vì lấy datetime.now() gây sai lệch
            continue

        day_idx = dt.weekday()  # 0: Thứ 2 ... 6: Chủ nhật
        hour = dt.hour

        if 0 <= hour < 12:
            time_idx = 0  # Sáng
        elif 12 <= hour < 18:
            time_idx = 1  # Chiều
        else:
            time_idx = 2  # Tối

        matrix[time_idx][day_idx] += 1

    return {
        "matrix": matrix,
        "days": ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"],
        "hours": ["Sáng", "Chiều", "Tối"]
    }


@app.get("/api/trend")
async def get_trend():
    """Thống kê Trend THẬT nhóm theo Ngày từ Database"""
    feedbacks = get_feedback()
    trend_map = {}

    for fb in feedbacks:
        created_str = fb.get("created_at", "")
        try:
            date_key = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S").strftime("%d/%m")
        except Exception:
            date_key = datetime.now().strftime("%d/%m")

        if date_key not in trend_map:
            trend_map[date_key] = {"positive": 0, "negative": 0, "neutral": 0}

        sentiment = parse_sentiment(fb.get("sentiment", ""))
        if sentiment == "Tích cực":
            trend_map[date_key]["positive"] += 1
        elif sentiment == "Tiêu cực":
            trend_map[date_key]["negative"] += 1
        else:
            trend_map[date_key]["neutral"] += 1

    # Sắp xếp chuẩn các ngày theo chuỗi ngày/tháng
    sorted_dates = sorted(
        trend_map.keys(),
        key=lambda x: datetime.strptime(x, "%d/%m")
    )

    return {
        "dates": sorted_dates if sorted_dates else [datetime.now().strftime("%d/%m")],
        "positive": [trend_map[d]["positive"] for d in sorted_dates] if sorted_dates else [0],
        "negative": [trend_map[d]["negative"] for d in sorted_dates] if sorted_dates else [0],
        "neutral": [trend_map[d]["neutral"] for d in sorted_dates] if sorted_dates else [0]
    }


@app.get("/api/faculty")
async def get_faculty_summary():
    feedbacks = get_feedback()
    faculty_data = {}

    for fb in feedbacks:
        fac = fb.get("faculty", "CNTT & TT")
        if fac not in faculty_data:
            faculty_data[fac] = {"total": 0, "positive": 0, "negative": 0, "neutral": 0}

        faculty_data[fac]["total"] += 1
        s = parse_sentiment(fb.get("sentiment", ""))
        if s == "Tích cực":
            faculty_data[fac]["positive"] += 1
        elif s == "Tiêu cực":
            faculty_data[fac]["negative"] += 1
        else:
            faculty_data[fac]["neutral"] += 1

    return faculty_data


@app.get("/api/report")
async def get_report_summary():
    feedbacks = get_feedback()
    total = len(feedbacks)
    return {
        "total_feedbacks": total,
        "resolved": sum(1 for fb in feedbacks if fb.get("status") in ["Đã xử lý", "Hoàn thành"]),
        "pending": sum(1 for fb in feedbacks if fb.get("status") not in ["Đã xử lý", "Hoàn thành"]),
        "export_date": datetime.now().strftime("%d/%m/%Y %H:%M")
    }


@app.get("/feedback")
async def feedback():
    return get_feedback()


@app.get("/statistics-data")
async def statistics_data():
    return get_statistics()


@app.put("/feedback/{id}")
async def change_status(id: int, data: dict):
    update_status(id, data.get("status", "Chưa xử lý"))
    invalidate_caches()
    return {"success": True}


@app.delete("/feedback/{id}")
async def remove_feedback(id: int):
    delete_feedback(id)
    invalidate_caches()
    return {"success": True}