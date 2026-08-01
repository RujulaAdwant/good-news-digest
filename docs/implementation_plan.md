# Good News Digest — Implementation Plan
**Author:** Rujula | **Deadline:** End of July 2026

---

## Overview

The pipeline in order:
```
Fetch → Store → Deduplicate → Score Sentiment → Summarize → Compile → Email
```
Build and test each stage independently before connecting them.

---

## Phase 0 — Setup (Do before Weekend 1)

**Tasks**
- [ ] Create GitHub repo, push empty project skeleton
- [ ] Set up Python 3.11+ virtual environment
- [ ] Install PostgreSQL locally
- [ ] Create accounts: TheNewsAPI, SendGrid, Anthropic
- [ ] Add `.env` file with all API keys
- [ ] Add `.env` to `.gitignore`
- [ ] Create `.cursorrules` file in project root
- [ ] Run `pip install` for all dependencies (sentence-transformers downloads model weights — do this early)

**Project structure**
```
good-news-digest/
├── app/
│   ├── main.py          # FastAPI app
│   ├── fetcher.py       # RSS + TheNewsAPI ingestion
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
```

**Dependencies**
```
fastapi
uvicorn
psycopg2-binary
feedparser
httpx
sentence-transformers
transformers
anthropic
sendgrid
apscheduler
python-dotenv
pytest
```

**Learning resources for setup**
- FastAPI docs — Getting Started: https://fastapi.tiangolo.com/tutorial/
- PostgreSQL + psycopg2 basics: https://www.psycopg.org/docs/usage.html
- python-dotenv: https://github.com/theskumar/python-dotenv

---

## Phase 1 — News Ingestion + Database (Weekend 1)

**Goal:** By end of weekend, you can run a script that fetches articles and stores them in PostgreSQL.

### Step 1.1 — Database schema
Write and run `schema.sql`:
```sql
CREATE TABLE articles (
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

CREATE TABLE digests (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    article_ids INTEGER[],
    email_sent_at TIMESTAMP
);
```

**Concepts to understand:** Primary keys, UNIQUE constraints (prevents duplicate URLs), SERIAL auto-increment, why we use TIMESTAMP vs DATE.

### Step 1.2 — TheNewsAPI integration
Write `fetcher.py` to call TheNewsAPI and return a list of article dicts.

Test with a small batch first — fetch 10 articles, print titles to terminal, confirm it works before touching the database.

**Concepts to understand:** REST API calls, JSON parsing, API rate limits, why we store `published_at` rather than today's date.

### Step 1.3 — RSS feed parsing
Add RSS support to `fetcher.py` using `feedparser`. Pick 5 starter feeds:
- AP News: `https://feeds.apnews.com/rss/topnews`
- Good News Network: `https://www.goodnewsnetwork.org/feed/`
- Positive News: `https://www.positive.news/feed/`
- Reuters: `https://feeds.reuters.com/reuters/topNews`
- BBC: `http://feeds.bbci.co.uk/news/rss.xml`

**Concepts to understand:** What RSS/XML is, how feedparser normalizes different feed formats.

### Step 1.4 — Store to PostgreSQL
Write a `save_articles()` function that inserts fetched articles into the database, skipping any with a URL that already exists (`INSERT ... ON CONFLICT DO NOTHING`).

**Concepts to understand:** SQL INSERT, ON CONFLICT (upsert pattern), why URL is the right deduplication key at this stage.

### Step 1.5 — FastAPI skeleton + `/fetch` endpoint
Wrap the fetcher in a FastAPI endpoint so you can trigger it via HTTP:
```
POST /fetch → runs fetcher, stores articles, returns count
GET /articles → returns stored articles with optional filters
```

Run with `uvicorn app.main:app --reload` and test in the browser at `localhost:8000/docs`.

**Concepts to understand:** What an API framework does, path operations, Pydantic response models, why `/docs` works automatically.

**Learning resources**
- TheNewsAPI docs: https://www.thenewsapi.com/documentation
- feedparser docs: https://feedparser.readthedocs.io/
- SQL tutorial (if rusty): https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-insert/
- FastAPI path operations: https://fastapi.tiangolo.com/tutorial/first-steps/

---

## Phase 2 — NLP Pipeline (Week 2)

**Goal:** By end of week, fetched articles are deduplicated, sentiment-scored, and summarized.

### Step 2.1 — Deduplication with sentence-transformers
Write `deduplicator.py`:

1. Load the `all-MiniLM-L6-v2` model (downloads once, cached locally)
2. For each new batch of articles, encode titles into vectors (embeddings)
3. Compare each new article against existing articles using cosine similarity
4. Flag articles above 0.85 threshold as `is_duplicate = True`

Test by intentionally fetching the same story from two sources and confirming it gets flagged.

**Concepts to understand:** What an embedding is (a list of numbers representing meaning), why cosine similarity measures semantic closeness, what the 0.85 threshold means and how to tune it.

**Interview question you should be able to answer:** "Why use embeddings instead of just comparing headlines with string matching?"

### Step 2.2 — Sentiment classification
Write `sentiment.py` using HuggingFace:

```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english")
```

Run each article's title + first paragraph through the pipeline. Store the score (0-1) in `sentiment_score`. Start with a threshold of 0.6 — tune after seeing real results.

**Concepts to understand:** What a pretrained model is and why you don't need to train your own, what SST-2 fine-tuning means, why 0.6 is a starting point not a fixed answer.

### Step 2.3 — Summarization with Claude API
Write `summarizer.py`:

- Only summarize articles that passed sentiment filtering (save API costs)
- Prompt: "Summarize this news article in 2-3 sentences, focusing on what's hopeful or constructive"
- Set `max_tokens=150`
- Store result in `summary` column

**Concepts to understand:** Token limits and why they matter for cost, prompt design for consistent output, why you filter before summarizing.

**Learning resources**
- sentence-transformers quickstart: https://www.sbert.net/docs/quickstart.html
- HuggingFace pipelines: https://huggingface.co/docs/transformers/pipeline_tutorial
- Cosine similarity explained simply: https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db
- Anthropic API docs: https://docs.anthropic.com/en/api/getting-started

---

## Phase 3 — Digest + Delivery (Week 3)

**Goal:** A formatted digest email lands in your inbox every morning automatically.

### Step 3.1 — Digest compiler
Write `digest.py`:

- Query PostgreSQL for today's articles where `is_duplicate = FALSE` and `sentiment_score >= 0.6`
- Order by sentiment score descending
- Select top 8-10
- Create a digest record in the `digests` table

### Step 3.2 — Email formatting
Format the digest as clean HTML email:
- Subject: "Your Good News Digest — [date]"
- Each story: headline (linked to article), source, 2-3 sentence summary
- Keep it simple — plain layout, readable on mobile

### Step 3.3 — SendGrid delivery
Write `emailer.py` to send the formatted email via SendGrid API. Test by sending to yourself manually before wiring up the scheduler.

### Step 3.4 — APScheduler
Write `scheduler.py` to run the full pipeline daily at 7am:

```python
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

@scheduler.scheduled_job('cron', hour=7)
def daily_digest():
    fetch_articles()
    deduplicate()
    score_sentiment()
    summarize()
    compile_and_send()

scheduler.start()
```

Test by setting the time to 2 minutes from now, confirming it runs, then changing back to 7am.

**Concepts to understand:** What a cron job is, blocking vs background schedulers, what happens if a job fails mid-pipeline (error handling).

**Learning resources**
- SendGrid Python quickstart: https://docs.sendgrid.com/for-developers/sending-email/quickstart-python
- APScheduler docs: https://apscheduler.readthedocs.io/en/3.x/userguide.html
- HTML email basics: https://www.litmus.com/blog/html-email-basics

---

## Phase 4 — Polish + Deploy (Final Days)

**Goal:** Live, deployed service you can demo and link on your resume.

### Step 4.1 — Error handling ✅
Wrap each pipeline stage in try/except so one failure doesn't crash everything. Log errors clearly.

Done in pipeline modules, `POST /process`, and `scheduler.run_daily_pipeline` (per-stage isolation).

### Step 4.2 — Deploy to Railway
- Push code to GitHub
- Create new Railway project, connect GitHub repo
- Add PostgreSQL plugin
- Add all environment variables in Railway dashboard
- Deploy

Scaffolding in-repo: `Dockerfile`, `railway.toml`, `python -m db.migrate`, `GET /health`.
Follow the **Deploy on Railway** section in `README.md` (API service + scheduler worker).

Railway is the simplest option for Python + PostgreSQL — use the Dockerfile so CPU torch / HF models install cleanly.

### Step 4.3 — Write README ✅
Your README is part of the portfolio. Include:
- What the project does and why (1 paragraph)
- Architecture diagram or pipeline description
- Tech stack with brief rationale
- How to run locally
- What you learned

### Step 4.4 — Tune and test
Run for 3-4 days. Ask yourself:
- Are the stories actually positive and interesting?
- Are duplicates being caught?
- Are summaries accurate?
- Does it run every morning without manual intervention?

Adjust sentiment threshold and source list based on real output.
See README **Tunables** table for env knobs.

**Learning resources**
- Railway deployment: https://docs.railway.app/getting-started
- Writing a good README: https://www.makeareadme.com/
- Python logging: https://docs.python.org/3/howto/logging.html

---

## Key Concepts to Master (Interview Prep)

For each of these, you should be able to explain it in 2 minutes without notes:

| Concept | Why it matters |
|---|---|
| What an embedding is | Core to your deduplication approach |
| Cosine similarity | How deduplication actually works |
| Pretrained vs fine-tuned models | Why you didn't train from scratch |
| Sentiment classification threshold tuning | Shows you understand ML isn't binary |
| Why FastAPI over Flask | Shows you made an intentional decision |
| Token limits and API costs | Shows production awareness |
| Cron scheduling | Fundamental backend concept |
| ENV variables and secrets management | Table stakes for any engineering job |

---

## Weekly Checklist

**Before Weekend 1**
- [ ] All accounts created, API keys in .env
- [ ] Dependencies installed (especially sentence-transformers)
- [ ] Empty project structure on GitHub

**End of Weekend 1**
- [ ] Can fetch articles from TheNewsAPI and 3+ RSS feeds
- [ ] Articles stored in PostgreSQL
- [ ] `/fetch` and `/articles` endpoints working
- [ ] Can see articles in database via psql or a DB viewer

**End of Week 2**
- [ ] Deduplication running and catching real duplicates
- [ ] Sentiment scores in database
- [ ] Claude summaries generating and storing correctly
- [ ] Full pipeline runnable manually end to end

**End of Week 3**
- [ ] Digest email landing in inbox manually triggered
- [ ] APScheduler running pipeline automatically
- [ ] No crashes on bad input or API failures

**End of July**
- [x] README written
- [ ] Deployed on Railway
- [ ] Running autonomously for 3+ days
- [ ] Link on resume/portfolio