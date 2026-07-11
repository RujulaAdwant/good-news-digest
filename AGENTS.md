# Good News Digest

A backend-only daily news digest service. Pipeline: Fetch → Store → Deduplicate → Score Sentiment → Summarize → Compile → Email. See `implementation_plan.md` for the full design and `.cursorrules` for code conventions.

## Cursor Cloud specific instructions

### Project state
- This repo is currently a **Phase 0 skeleton**: `app/*.py`, `scheduler.py`, and `db/schema.sql` are empty (0 bytes). There is no runnable FastAPI app, scheduler, DB schema, or test suite yet. `uvicorn app.main:app` and `python scheduler.py` will fail until those files are implemented. The intended schema DDL lives in `implementation_plan.md` (§1.1).

### Python environment
- Dependencies are installed into a virtualenv at `.venv` (gitignored, so it never ships with the repo — each fresh VM recreates it via the update script). Activate with `source .venv/bin/activate`, or call binaries directly, e.g. `.venv/bin/python`, `.venv/bin/uvicorn`, `.venv/bin/pytest`.
- Target runtime is Python 3.11+ (VM has 3.12). `torch`/`transformers`/`sentence-transformers` are heavy; the first dependency install is large.
- Gotcha: `python-dotenv`'s `load_dotenv()` (no args) raises `AssertionError` when a script is piped via a heredoc/stdin (`python - <<'PY'`) because it can't introspect the caller frame. Pass an explicit path (`load_dotenv("/workspace/.env")`) or run from a real file.

### PostgreSQL
- PostgreSQL 16 is installed but **not started on boot**. Start it with: `sudo pg_ctlcluster 16 main start` (check with `sudo -u postgres pg_lsclusters`).
- A dev database and role are provisioned: connect via `postgresql://digest:digest@localhost:5432/good_news_digest`.

### Env vars
- `.env` is gitignored. A local dev `.env` exists with `DATABASE_URL` (pointing at the dev DB) and the tunable thresholds. External-service keys (`ANTHROPIC_API_KEY`, `NEWSAPI_KEY`, `SENDGRID_API_KEY`, `SENDGRID_FROM_EMAIL`, `DIGEST_RECIPIENT_EMAIL`) are blank — fill them in to exercise NewsAPI, Claude, or SendGrid.

### ML model weights
- HuggingFace models (`all-MiniLM-L6-v2`, `distilbert-base-uncased-finetuned-sst-2-english`) download on first use and cache to `~/.cache/huggingface`. First run needs network egress; subsequent runs are offline-capable.
