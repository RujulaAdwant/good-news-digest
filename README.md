# The Glass Digest

A daily email digest that cuts through sensational news cycles to surface constructive, solutions-oriented stories.

## The Why
What's the first thing you do when you wake up in the morning? I wish I could tell you that I open my eyes at the first ring my old-school alarm clock, straight out of a 2000's movie, and jump headfirst into seizing the day.

My mornings typically look more like reaching for the Snooze or Dismiss button on my phone's alarm app, and once awake, immediately checking my email to catch up on any messages I missed. Often, in an attempt to replace the social media dopamine craving first thing in the morning, I tap into my news apps, only to feel overwhelmed by an abundance of information and the headlines' striking negativity.

Headlines are sensational by design. They are meant, like so many other agents of influence, to capture our attention spans in the hopes that we let our mind linger on an idea for a bit before moving onto the next shiny thing that catches our eye. 

When I open my news apps in the morning, I want my attention to be diverted to the issues and events that matter. I consistently achieve this goal by scrolling through top headlines. However, the sense of mental fatigue that I face upon confronting predictions of impending doom isn't always what I signed up for.

The Glass Digest isn't trying to replace real journalism or the hard truths it covers. It's a way to start my day as a curious student of the world, learning about what's happening through stories that inform without provoking dread.

## Pipeline

```text
Fetch (TheNewsAPI + RSS)
  → Store (PostgreSQL, UNIQUE url)
  → Deduplicate (sentence-transformers cosine similarity)
  → Score sentiment (DistilBERT SST-2)
  → Filter relevance (good-news vs corporate/horoscope prototypes)
  → Summarize digest picks (Claude Sonnet)
  → Compile daily digest row
  → Email via SendGrid
```

Selection prefers high-sentiment, topic-diverse stories (`DIGEST_SIZE`, default 5). Usable Claude summaries are included when present; stories without a summary still appear as title + source.

## Tech stack

| Piece | Why |
|---|---|
| **FastAPI** | Typed request/response models, automatic OpenAPI at `/docs` |
| **PostgreSQL + psycopg2** | Durable article/digest store; `ON CONFLICT DO NOTHING` on URL |
| **sentence-transformers (`all-MiniLM-L6-v2`)** | Cheap semantic embeddings for near-duplicate detection and topic diversity |
| **DistilBERT SST-2** | Off-the-shelf positive/negative scores without training a classifier |
| **Claude (`claude-sonnet-4-6`)** | Short hopeful summaries only for selected picks (cost-aware) |
| **SendGrid** | Transactional HTML email |
| **APScheduler** | Daily cron at `DIGEST_HOUR` in `DIGEST_TIMEZONE` |

## API (manual runs)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/fetch` | Ingest TheNewsAPI + RSS |
| `POST` | `/process` | Dedup → sentiment → summarize |
| `POST` | `/digest` | Compile today's digest (no email) |
| `POST` | `/send` | Email the compiled digest |
| `DELETE` | `/digest` | Unlock same-day recompile/resend (`ALLOW_DIGEST_RESET=true`) |
| `GET` | `/articles` | Inspect stored rows |

Interactive docs: `http://127.0.0.1:8000/docs`

## Local setup

**Prerequisites:** Python 3.11+, PostgreSQL, API keys for Anthropic, TheNewsAPI, and SendGrid.

```bash
git clone git@github.com:RujulaAdwant/good-news-digest.git
cd good-news-digest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill DATABASE_URL, THENEWSAPI_KEY, ANTHROPIC_API_KEY,
# SENDGRID_*, DIGEST_RECIPIENT_EMAIL

createdb good_news_digest   # or use the URL in .env
python -m db.migrate

./run.sh                   # API with reload on :8000
# in another terminal:
python scheduler.py        # daily pipeline worker
```

End-to-end smoke (without waiting for cron):

```bash
curl -X POST http://127.0.0.1:8000/fetch
curl -X POST http://127.0.0.1:8000/process
curl -X POST http://127.0.0.1:8000/digest
curl -X POST http://127.0.0.1:8000/send
```

Tests (external APIs mocked):

```bash
pytest -q
```

## Deploy on Railway

You need **two services** from the same repo (API + scheduler) plus a Postgres plugin.

1. Push this repo to GitHub (already: `RujulaAdwant/good-news-digest`).
2. [Railway](https://railway.app) → New Project → Deploy from GitHub.
3. Add a **PostgreSQL** plugin; copy its `DATABASE_URL` into the app service.
4. **API service**
   - Builder: Dockerfile (see `railway.toml`)
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
5. **Worker service** (same repo)
   - Same Dockerfile / env
   - Start: `python scheduler.py`
6. Set env vars on **both** services (from `.env.example`):

   `DATABASE_URL`, `THENEWSAPI_KEY`, `ANTHROPIC_API_KEY`, `SENDGRID_API_KEY`,
   `SENDGRID_FROM_EMAIL`, `DIGEST_RECIPIENT_EMAIL`, plus optional tunables
   (`SIMILARITY_THRESHOLD`, `SENTIMENT_THRESHOLD`, `DIGEST_SIZE`,
   `TOPIC_DIVERSITY_THRESHOLD`, `DIGEST_HOUR`, `DIGEST_TIMEZONE`,
   `FETCH_WINDOW_HOURS`, `DIGEST_BANNER_URL`). Keep `ALLOW_DIGEST_RESET=false`
   in production.

7. One-time schema: Railway shell / one-off run:

   ```bash
   python -m db.migrate
   ```

**Note:** First boot downloads HuggingFace models (`all-MiniLM-L6-v2`, DistilBERT). Prefer a plan with enough RAM/disk; the Dockerfile installs CPU PyTorch to avoid CUDA wheels.

## Tunables

Starting points — adjust after a few real digests:

| Env | Default | Role |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.85` | Near-duplicate cosine cutoff |
| `SENTIMENT_THRESHOLD` | `0.6` | Minimum positive score to enter the pool |
| `SENTIMENT_TARGET` | *(empty)* | Prefer scores near this (Glass); empty = max-positive |
| `TOPIC_DIVERSITY_THRESHOLD` | `0.70` | Same-topic rejection in digest picks |
| `RELEVANCE_MARGIN` | `0.05` | Good-news vs corporate prototype margin |
| `DIGEST_SIZE` | `5` | Stories per email |
| `DIGEST_HOUR` / `DIGEST_TIMEZONE` | `7` / `America/Los_Angeles` | Cron |
| `DIGEST_BANNER_URL` | *(empty)* | Reserved for future email header banner |

## What I learned

- Separating pipeline stages (fetch / NLP / compile / email) keeps failures isolated and the API thin.
- Semantic dedup is necessary beyond URL uniqueness — wire copy often lands under different links.
- Thresholds are product decisions: too strict and the digest is empty; too loose and you get near-dupes or fluff.
- Summarize only the final picks, cache in Postgres, and never surface model refusals as “summaries.”
- Deploying embedding models means treating cold start, image size, and CPU torch as first-class constraints.

## Project layout

See [`INDEX.md`](INDEX.md). Design walkthrough and interview notes live under [`docs/`](docs/).

## Demo {coming soon!}

## Future improvements
- Digest history web UI — a lightweight React frontend to browse past digests without hitting the API directly
- Feedback loop — thumbs up/down per article, feeding into per-source or per-topic filtering over time
- Topic clustering — group digest picks by theme (e.g. science, climate, innovation) using the same embeddings already computed for deduplication
- Multi-user support — Google OAuth + per-user source preferences and delivery times
- Fine-tuned sentiment model — once enough feedback data exists, move beyond off-the-shelf DistilBERT toward a model tuned specifically on constructive-news examples

## License
MIT
