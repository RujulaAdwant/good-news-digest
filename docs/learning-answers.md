# Interview Answers — Good News Digest

Answer key for `docs/learning-questions.md`.
Aim to explain each in ~1–2 minutes without notes. Wording can differ; the idea should match.

---

## Phase 1 — News Ingestion + Database

### `db/schema.sql`

1. **URL UNIQUE, title not.** A URL identifies one published page. Titles collide (rewrites, wire copy, generic headlines). UNIQUE on title would reject distinct articles or miss re-fetches of the same link.

2. **TIMESTAMP vs DATE.** `published_at` needs time-of-day for the fetch window and ordering. `digest_date` is “which calendar day’s digest,” not a moment.

3. **SERIAL.** PostgreSQL shorthand for an integer column backed by a sequence (`DEFAULT nextval(...)`). Ids auto-increment; you don’t assign them.

4. **Pre-create NLP columns.** Avoid schema churn later. Phase 1 leaves them NULL/default; Phase 2+ fills them. One table from day one.

5. **`articles.url` vs `digests.date`.** At most one row per article URL; at most one digest record per calendar day.

### `app/config.py`

6. **Env vars for secrets.** Don’t commit keys. Same code runs locally and in deploy with different env.

7. **Required DB, optional news key.** No database → nothing works. Empty news API key → skip Event Registry; RSS-only fetch still works.

8. **`@lru_cache`.** Caches the first `Settings` so env isn’t re-parsed every call. Clear with `get_settings.cache_clear()` after env changes or between tests.

9. **`frozen=True`.** Immutable settings — no accidental mid-request mutation. Change config by restarting or rebuilding settings.

10. **New tunable.** Add a field with a **default** (`os.getenv` / `_int_env` / `_float_env`), document in `.env.example`. Old deploys keep working until they set the var.

### `db/connection.py`

11. **Context managers.** Guaranteed cleanup: commit/rollback and `close()` even on errors. No leaked connections.

12. **Commit vs rollback.** Commit persists the transaction. Rollback undoes uncommitted work after a DB error.

13. **Catch `psycopg2.Error`.** Only DB failures take the rollback + “transaction failed” path. Other bugs aren’t mislabeled (though `finally` still closes).

14. **Non-DB exception.** The `except psycopg2.Error` block doesn’t run; `finally` still closes the connection. Transaction is typically aborted on close.

15. **Per-request vs pool.** Per-request (current): simple, fine at low volume. Pool: reuses connections under concurrency; more setup, better when open/close is a bottleneck.

### `db/articles.py`

16. **ON CONFLICT.** Insert attempted → if `url` already exists (UNIQUE violation), Postgres skips the insert instead of erroring. No row update with `DO NOTHING`.

17. **RETURNING id.** Successful insert returns an id; conflict returns no row → `fetchone()` is `None` → count as skipped. Clean inserted vs skipped without a separate SELECT.

18. **URL as storage dedup key.** Stable, available from every source, cheap. Titles diverge for the same story; body matching is NLP (embeddings).

19. **`.format()` for WHERE.** Safe *here*: only fixed clause text (`source = %s` or empty); values stay in bound params. Unsafe if you interpolated user strings into SQL.

20. **Dataclass vs Pydantic.** Internal DTO for fetcher ↔ DB — light, no HTTP validation. Pydantic belongs at the API boundary (`schemas.py`).

21. **Loop inserts.** Per-row `RETURNING` makes insert/skip counting simple. Batch/`COPY` is faster; conflict accounting is messier. Fine for Phase 1 batch sizes.

22. **NULLS LAST.** Newest first; missing `published_at` sinks to the bottom instead of sorting unpredictably.

### `app/fetcher.py`

23. **`fetch_articles` returns / doesn’t.** Returns filtered `list[ArticleRecord]`. Does **not** write to Postgres — `save_articles` / `main` does that.

24. **Event Registry + RSS.** Broader category coverage and structured JSON. RSS is simple/free but feed-limited. Together: coverage + resilience if one path fails.

25. **Continue on partial failure.** Partial success > total failure. One bad feed shouldn’t block the whole ingest.

26. **RSS/Atom + feedparser.** XML syndication formats. Field names and dates differ by feed; feedparser normalizes to `.entries` with `.title`, `.link`, `.published_parsed`, etc.

27. **`published_parsed`.** Already a `struct_time` — fewer format bugs. Raw strings vary and need fallback parsers.

28. **Store source `published_at`.** Story time drives window filter, ranking, freshness. Insert time ≠ publish time.

29. **Reject missing dates.** Can’t know if it’s in-window; keeping them risks stale/undated junk.

30. **Pacific then UTC.** Digest is “last N hours for a Pacific user.” Local wall-clock matters; store/compare in UTC for consistency.

31. **`seen_urls` vs DB UNIQUE.** In-memory: same URL twice in one run (multi-category/feed). DB: across runs / concurrent requests. Neither catches different URLs for the same story.

32. **`text_for_nlp` early.** Shared title+first-paragraph shape for later embeddings/sentiment — one helper, no rework in Phase 2.

33. **`dateStart` + client window.** API filter is coarse (calendar day). Client filter is the exact hour window in digest TZ and also gates RSS (no `dateStart`).

### `app/schemas.py`

34. **Pydantic over dicts.** Typed contract, serialization, docs, validation. Stable JSON shape for clients and OpenAPI.

35. **`response_model=`.** Validates/filters outgoing data to the model; generates OpenAPI schema for `/docs`.

36. **Record vs Response.** Different layers: DB/pipeline vs public API. API can evolve without rewriting DB types. Response requires persisted fields like `id` / `created_at`.

37. **NULL `is_duplicate`.** Column defaults to FALSE, so rows should be false in practice. Constructing without the field uses the Pydantic default `False`; a true SQL NULL would be a data bug to fix.

### `app/main.py`

38. **FastAPI vs handler.** Framework: routing, query parsing, status codes, serialization, docs. Handler: call fetch + save, map failures to 500, build `FetchResponse`.

39. **201 on `/fetch`.** Creates new article rows (resource creation). 200 = OK; 201 = created.

40. **No SQL in `main`.** Separation of concerns — thin routes; DB reusable by scheduler later; easier tests.

41. **`/docs`.** FastAPI builds OpenAPI from path ops, type hints, `response_model`, Query constraints, docstrings → Swagger UI.

42. **Sync routes.** Work is blocking (psycopg2, sync HTTP, feedparser). `async def` without async I/O doesn’t help and can block the event loop.

43. **`fetched=20, inserted=0, skipped=20`.** Twenty in-window articles; all URLs already in DB. Normal on re-run — not an error.

44. **Auth later.** API key / Bearer dependency, or keep the trigger private (network restriction). Don’t leave an open fetch trigger on the public internet.

### Phase 1 — Tests & env

45. **Mock APIs, real DB.** External APIs: flaky, keyed, rate-limited. ON CONFLICT behavior is what you’re verifying — needs Postgres.

46. **API-only gap.** Wrong SQL, broken UNIQUE/ON CONFLICT, param bugs, commit issues — mocked `save_articles` never exercises them.

47. **`cache_clear`.** `@lru_cache` keeps first Settings; without clear, one test’s settings bleed into the next.

48. **`.env.example` vs `.env`.** Documents required keys; real secrets stay local / deploy secrets.

49. **Blank for Phase 1.** Anthropic, SendGrid, recipient unused yet. Empty news key → RSS-only. Need a real `DATABASE_URL`.

### Phase 1 — Cross-cutting

50. **Trace.** Event Registry JSON → parse to `ArticleRecord` → optional window keep → `save_articles` INSERT → `GET /articles` → `ArticleResponse` JSON.

51. **Three layers.** (1) In-batch `seen_urls` — misses cross-run / different URLs. (2) `UNIQUE url` + ON CONFLICT — misses same story, different URLs. (3) Fetch window — not dedup; drops old/undated. Phase 2 embeddings cover semantic duplicates.

52. **ER down, RSS up.** ER errors logged; RSS still runs; `/fetch` 201 with whatever RSS (and window) produced.

53. **Embedding dedup home.** `app/deduplicator.py`; after store; update `is_duplicate`. Orchestrated by `/process` or scheduler — no SQL in fetcher.

54. **FastAPI vs Flask.** Native type hints + Pydantic → validation and free OpenAPI/`/docs`. Fits an API-first service without extra Flask plugins.

---

## Phase 2 — NLP Pipeline

### Deduplication

55. **Embedding.** A list of numbers representing the *meaning* of a piece of text in a vector space, so similar meanings land near each other.

56. **Why not string matching.** Reworded headlines of the same story share meaning but not exact/near-exact strings. Embeddings catch semantic near-duplicates.

57. **Cosine similarity.** Measures angle between vectors (alignment of meaning). After L2-normalize, it’s a dot product. **0.85** means “very similar direction” — treated as the same story if ≥ threshold.

58. **Tunable threshold.** Too low → false duplicates (over-flag). Too high → miss real dupes. Tune on real pairs from your sources.

59. **Keep older.** Lower `id` (A) stays canonical; B gets `is_duplicate=True`.

60. **Window only.** Daily digest cares about recent stories; full-table compare grows O(n²)-ish expensive and may match unrelated old stories.

61. **Title + paragraph.** Richer signal when headlines diverge but lead paragraphs describe the same event.

62. **Load once.** Model load is heavy (disk/GPU/CPU). Per-call reload would dominate latency. Lazy singleton: load on first use, reuse afterward.

63. **URL vs semantic.** Same URL → same page (storage). Same *story*, different publishers/URLs → only embeddings catch it.

### Sentiment

64. **Pretrained.** Trained on large corpora by others; you download weights. Training from scratch needs huge labeled data and compute you don’t need for v1.

65. **Fine-tuned on SST-2.** Base DistilBERT further trained on Stanford Sentiment Treebank (positive/negative). Task-specific head on top of general language understanding.

66. **NEGATIVE @ 0.75.** Store `1 - 0.75 = 0.25`. That does **not** pass a 0.6 positive threshold.

67. **Single float.** One comparable score for thresholding and sorting (“most positive first”) without a second column.

68. **Tunable 0.6.** News text ≠ movie reviews (SST-2 domain). Starting point — adjust when output quality is wrong.

69. **Skip dupes / already scored.** Don’t waste compute on discarded copies; don’t overwrite existing scores on re-run (`IS NULL` = idempotent).

### Summarization

70. **Filter before Claude.** API calls cost money and latency. Only summarize stories that will appear in the digest.

71. **Cache summary.** Idempotent `/process`; never re-pay for the same article. `summary IS NULL` is the gate.

72. **`max_tokens`.** Caps generated output length (and thus cost). 150 is enough for 2–3 sentences.

73. **Retries.** Transient network/API errors are common; 2 retries with backoff recover without failing the whole stage for one blip.

74. **Missing key.** Log warning, skip summarization, return 0 — same pattern as optional news API key.

### Orchestration

75. **`/process`.** Dedup windowed articles → score unscored non-dupes → summarize eligible positive non-dupes without summaries → return counts + any `stage_errors`.

76. **Continue after failure.** One stage failing shouldn’t kill the pipeline (cursorrules). Record error in `stage_errors`; later stages still run on whatever data is ready.

77. **Separate `/process`.** Fetch and NLP are different concerns; easier to debug/re-run NLP without re-hitting external news APIs. Scheduler can call both in order later.

78. **Summary eligibility.** In fetch window AND `is_duplicate = FALSE` AND `sentiment_score >= threshold` AND `summary IS NULL`.

### Phase 2 — Cross-cutting

79. **Order.** Fetch → Store (cheap URL dedup) → semantic dedup → sentiment → summarize → (later) compile/email. Each stage narrows the set before more expensive work.

80. **Tuning.** Run for days; spot-check false dupes / missed dupes / “positive” stories that aren’t; adjust `SIMILARITY_THRESHOLD` and `SENTIMENT_THRESHOLD` in env; re-run `/process` on pending rows as needed.

81. **Storage vs NLP dedup.** URL: exact same link. Embeddings: same meaning, different links/wording.

82. **Testing NLP.** Unit-test pure helpers (`positive_probability`, `find_duplicate_ids` with mocked embeddings). Mock HuggingFace pipeline and Anthropic client. Never hit real Claude/News APIs in CI.

---

## Not yet implemented

Phase 3–4 answers will be added when digest, email, scheduler, and deploy land.
