## Folder structure
good-news-digest/
├── app/
│   ├── main.py          # FastAPI app
│   ├── fetcher.py       # RSS + NewsAPI ingestion
│   ├── deduplicator.py  # Embedding similarity
│   ├── sentiment.py     # Sentiment scoring
│   ├── summarizer.py    # Claude API calls
│   ├── digest.py        # Digest compiler
│   └── emailer.py       # SendGrid delivery
├── db/
│   └── schema.sql       # PostgreSQL schema
├── scheduler.py         # APScheduler jobs
├── .env                 # API keys — never commit
├── .gitignore
├── requirements.txt
└── README.md