## Folder structure

```text
good-news-digest/
├── app/
│   ├── main.py           # FastAPI routes
│   ├── config.py         # Env-backed settings
│   ├── fetcher.py        # RSS + TheNewsAPI ingestion
│   ├── deduplicator.py   # Embedding near-duplicate detection
│   ├── sentiment.py      # DistilBERT SST-2 scoring
│   ├── relevance.py      # Good-news vs corporate/horoscope filter
│   ├── summarizer.py     # Claude summaries for digest picks
│   ├── digest.py         # Rank, diversify, compile digests
│   ├── emailer.py        # HTML format + SendGrid send
│   ├── schemas.py        # Pydantic API models
│   └── exceptions.py     # Domain errors
├── db/
│   ├── schema.sql        # PostgreSQL DDL
│   ├── migrate.py        # Idempotent schema apply
│   ├── articles.py       # Article queries
│   └── digests.py        # Digest row queries
├── tests/                # Pytest suite (external APIs mocked)
├── docs/                 # Plan, notes, learning Q&A
├── images/               # Local banner assets (not used in email yet)
├── scheduler.py          # APScheduler daily pipeline
├── Dockerfile            # Railway / CPU torch image
├── railway.toml          # API service defaults
├── run.sh                # Local uvicorn helper
├── .env                  # Secrets — never commit
├── .env.example          # Documented env vars
├── .gitignore
├── requirements.txt
└── README.md
```
