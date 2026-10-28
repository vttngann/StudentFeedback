SELECT TOP 1 *
FROM feedback;

SELECT TOP 10 id, created_at
FROM feedback
ORDER BY id DESC;

sp_help feedback

ALTER TABLE feedback
ADD faculty NVARCHAR(100) NULL;

ALTER TABLE feedback
ADD priority NVARCHAR(20) NULL;

DELETE FROM feedback;
DBCC CHECKIDENT ('feedback', RESEED, 0);