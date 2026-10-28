import pyodbc

# ======================================
# SQL Server Config
# ======================================

SERVER = r"localhost\VTTNGAN"
DATABASE = "StudentFeedback"

CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
)


# ======================================
# Kết nối SQL Server
# ======================================

def get_connection():
    return pyodbc.connect(CONNECTION_STRING)


# ======================================
# Khởi tạo Database
# ======================================

def init_db():
    # Đã tạo bảng bằng SQL Server nên không cần tạo nữa
    pass


# ======================================
# Lưu phản hồi
# ======================================

def save_feedback(comment, result):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback
        (
            comment,
            faculty,
            sentiment,
            emotion,
            topic,
            priority,
            status,
            sentiment_confidence,
            topic_confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
    """,
    (
        comment,
        result["faculty"],
        result["sentiment"],
        result["emotion"],
        result["topic"],
        result["priority"],
        "Chưa xử lý",
        result["sentiment_confidence"],
        result["topic_confidence"]
    ))

    conn.commit()
    conn.close()


# ======================================
# Lấy toàn bộ phản hồi
# ======================================

def get_feedback():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM feedback
        ORDER BY id DESC
    """)

    columns = [column[0] for column in cursor.description]

    rows = []

    for row in cursor.fetchall():
        rows.append(dict(zip(columns, row)))

    conn.close()

    return rows


# ======================================
# Lấy 1 phản hồi
# ======================================

def get_feedback_by_id(id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM feedback
        WHERE id=?
    """, id)

    row = cursor.fetchone()

    if row is None:
        conn.close()
        return None

    columns = [column[0] for column in cursor.description]

    result = dict(zip(columns, row))

    conn.close()

    return result


# ======================================
# Dashboard Statistics
# ======================================

def get_statistics():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM feedback")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM feedback WHERE sentiment=N'Tích cực'")
    positive = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM feedback WHERE sentiment=N'Tiêu cực'")
    negative = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM feedback WHERE sentiment=N'Trung tính'")
    neutral = cursor.fetchone()[0]

    cursor.execute("""
        SELECT topic,
               COUNT(*) AS total
        FROM feedback
        GROUP BY topic
    """)

    topic_columns = [c[0] for c in cursor.description]
    topics = [
        dict(zip(topic_columns, row))
        for row in cursor.fetchall()
    ]

    cursor.execute("""
        SELECT sentiment,
               COUNT(*) AS total
        FROM feedback
        GROUP BY sentiment
    """)

    sentiment_columns = [c[0] for c in cursor.description]
    sentiments = [
        dict(zip(sentiment_columns, row))
        for row in cursor.fetchall()
    ]

    cursor.execute("""
        SELECT
            CONVERT(date, created_at) AS date,
            COUNT(*) AS total
        FROM feedback
        GROUP BY CONVERT(date, created_at)
        ORDER BY CONVERT(date, created_at)
    """)

    daily_columns = [c[0] for c in cursor.description]
    daily = [
        dict(zip(daily_columns, row))
        for row in cursor.fetchall()
    ]

    conn.close()

    return {

        "total": total,

        "positive": positive,

        "negative": negative,

        "neutral": neutral,

        "topics": topics,

        "sentiments": sentiments,

        "daily": daily

    }


# ======================================
# Cập nhật trạng thái
# ======================================

def update_status(id, status):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE feedback
        SET status=?
        WHERE id=?
    """, status, id)

    conn.commit()
    conn.close()


# ======================================
# Xóa phản hồi
# ======================================

def delete_feedback(id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM feedback
        WHERE id=?
    """, id)

    conn.commit()
    conn.close()


# ======================================
# Xóa toàn bộ dữ liệu
# ======================================

def clear_feedback():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM feedback")

    conn.commit()
    conn.close()