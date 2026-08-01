CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    source TEXT,
    published_at TIMESTAMP,
    full_text TEXT,
    sentiment_score FLOAT,
    summary TEXT,
    is_duplicate BOOLEAN DEFAULT FALSE,
    digest_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS digests (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    article_ids INTEGER[],
    email_sent_at TIMESTAMP
);
