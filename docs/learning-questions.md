# Learning Questions — Good News Digest

Questions tied to each implemented phase. Use `docs/learning-answers.md` as the answer key.
Add new sections here when later phases ship.

---

## Phase 1 — News Ingestion + Database

### `db/schema.sql`

1. Why is `url` UNIQUE but `title` is not?
2. Why is `published_at` a TIMESTAMP and `digest_date` a DATE?
3. What does `SERIAL` expand to under the hood?
4. Why pre-create `sentiment_score` / `is_duplicate` / `summary` before those pipelines exist?
5. What's the difference between uniqueness on `articles.url` and on `digests.date`?

### `app/config.py`

6. Why put secrets in environment variables instead of constants in code?
7. Why is `database_url` required but `thenewsapi_key` allowed to be empty?
8. What does `@lru_cache` on `get_settings` do, and when would you clear it?
9. Why is `Settings` a `frozen=True` dataclass?
10. How would you add a new tunable config value without breaking existing deploys?

### `db/connection.py`

11. What problem do context managers solve for database connections?
12. What's the difference between commit and rollback in `get_connection`?
13. Why catch `psycopg2.Error` specifically instead of bare `Exception`?
14. What happens if the caller raises a non-DB exception inside the `with` block?
15. Connection-per-request vs a connection pool — tradeoffs for this service?

### `db/articles.py`

16. Walk through `INSERT ... ON CONFLICT (url) DO NOTHING`. What triggers the conflict?
17. Why use `RETURNING id` to count inserts instead of relying on `rowcount` alone?
18. Why is URL the right deduplication key at the *storage* layer (not title or body)?
19. Is building the `WHERE` clause via `.format()` safe here? When would it not be?
20. Why is `ArticleRecord` a dataclass in `db/` instead of a Pydantic model?
21. Why loop one insert at a time instead of `executemany` / `COPY`?
22. What does `ORDER BY published_at DESC NULLS LAST` buy you?

### `app/fetcher.py`

23. End-to-end: what does `fetch_articles()` return, and what does it *not* do?
24. Why use TheNewsAPI (HTTP JSON API) in addition to RSS?
25. Why continue on a single category/feed failure instead of failing the whole fetch?
26. What is RSS/Atom, and what does feedparser normalize for you?
27. Why prefer `published_parsed` over raw `published` date strings?
28. Why store source `published_at` instead of using “today” or insert time?
29. Why reject articles with `published_at is None`?
30. Why compute the fetch cutoff in `America/Los_Angeles` then convert to UTC?
31. What does the in-memory `seen_urls` set catch that DB `UNIQUE` does not (and vice versa)?
32. Why does `text_for_nlp` exist before sentiment/embeddings were implemented?
33. TheNewsAPI already gets `published_after` — why still filter with `within_fetch_window`?

### `app/schemas.py`

34. Why return Pydantic models from FastAPI instead of raw dicts?
35. What does `response_model=` do for validation *and* OpenAPI?
36. Why separate `ArticleRecord` (dataclass) from `ArticleResponse` (Pydantic)?
37. `is_duplicate: bool = False` — what happens if the DB somehow returned NULL?

### `app/main.py`

38. What does FastAPI handle for you vs what `trigger_fetch` still owns?
39. Why return HTTP 201 on `POST /fetch` instead of 200?
40. Why is SQL *not* in `main.py`?
41. How does `/docs` get generated from these decorators and types?
42. Why are these routes `def` (sync) rather than `async def`?
43. What would `fetched=20, inserted=0, skipped=20` mean? Is it an error?
44. How would you auth-protect `POST /fetch` if this were public?

### Phase 1 — Tests & env

45. Why mock external APIs in unit tests but hit a real DB for `test_db_articles`?
46. What bug would pass if you only had API tests and no DB integration tests?
47. Why clear `get_settings.cache_clear()` between tests?
48. Why commit `.env.example` but gitignore `.env`?
49. Which env vars can stay blank and still let you exercise Phase 1?

### Phase 1 — Cross-cutting

50. Trace one article from TheNewsAPI JSON → `ArticleRecord` → `INSERT` → `GET /articles` JSON.
51. Name three different “dedup / filter” layers in Phase 1 and what each misses.
52. If TheNewsAPI is down but RSS works, what does `POST /fetch` return?
53. Where would embedding-based dedup live without breaking Phase 1 layout?
54. Why is FastAPI a better fit than Flask for this project’s API layer?

---

## Phase 2 — NLP Pipeline

### Deduplication (`app/deduplicator.py`)

55. What is an embedding, in plain English?
56. Why use embeddings instead of string matching (equality / Levenshtein) on headlines?
57. What does cosine similarity measure, and what does a score of 0.85 mean here?
58. Why is 0.85 a *starting point*, not a fixed truth?
59. If articles A and B score 0.92 similarity and A has a lower `id`, which gets `is_duplicate=True`?
60. Why compare only within the fetch window, not the entire table?
61. Why embed title + first paragraph instead of title only?
62. Why load `all-MiniLM-L6-v2` once (lazy singleton) instead of on every call?
63. URL uniqueness already exists — why do you still need semantic dedup?

### Sentiment (`app/sentiment.py`)

64. What is a pretrained model, and why don’t you train sentiment from scratch?
65. What does “fine-tuned on SST-2” mean?
66. The model returns `NEGATIVE` with confidence `0.75`. What do we store in `sentiment_score`, and does it pass a 0.6 threshold?
67. Why store a single positive-probability float instead of keeping label + confidence separately?
68. Why is the sentiment threshold (0.6) tunable via env?
69. Why score only non-duplicates with `sentiment_score IS NULL`?

### Digest selection (`app/digest.py`)

70. Why select final digest picks *before* calling Claude, instead of summarizing every positive article?
71. Walk through greedy topic diversity: sort order, keep/skip rule, and when selection stops.
72. Why is `TOPIC_DIVERSITY_THRESHOLD` (default 0.70) lower than `SIMILARITY_THRESHOLD` (0.85)?
73. Why is `DIGEST_SIZE` (default 5) an env knob rather than a hardcoded constant?
74. What if fewer than `DIGEST_SIZE` candidates survive the diversity filter?
75. Why live in `digest.py` instead of inside `summarizer.py`?

### Summarization (`app/summarizer.py`)

76. After digest selection, which of those picks does Claude actually get called for?
77. Why cache `summary` in Postgres and never re-summarize the same article?
78. What does `max_tokens=150` control, and why does it matter for cost?
79. Why retry Claude calls (max 2 retries) instead of failing immediately?
80. What happens if `ANTHROPIC_API_KEY` is missing?
81. If digest selection fails, how does that show up in the summarize stage?

### Orchestration (`POST /process`)

82. Walk through what `POST /process` does, stage by stage.
83. If deduplication throws, do sentiment and summarization still run? Why design it that way?
84. Why a separate `/process` endpoint instead of folding NLP into `/fetch`?
85. Which articles get a Claude summary (list every gate, including digest selection)?

### Phase 2 — Cross-cutting / interview favorites

86. Explain the full pipeline order and why each stage comes before the next.
87. How would you tune similarity, sentiment, digest size, and topic diversity in production?
88. Storage-layer dedup (URL) vs NLP dedup (embeddings) vs digest topic diversity — what’s each for?
89. How do you test NLP / digest stages without downloading models or calling Claude in CI?

---

## Not yet implemented (placeholder)

Digest *selection* shipped early with Phase 2 summarization. Remaining Phase 3 work
(digest row / HTML email / SendGrid / APScheduler) and Phase 4 (deploy + polish) get
questions when those land.
