USE StudentFeedback;
GO

CREATE TABLE feedback (
    id INT IDENTITY(1,1) PRIMARY KEY,
    comment NVARCHAR(MAX) NOT NULL,
    sentiment NVARCHAR(50),
    emotion NVARCHAR(50),
    topic NVARCHAR(100),
    sentiment_confidence FLOAT,
    topic_confidence FLOAT,
    status NVARCHAR(50) DEFAULT N'Chưa xử lý',
    created_at DATETIME DEFAULT GETDATE()
);
GO

SELECT TOP 10 *
FROM feedback
ORDER BY id DESC;