# Bugs Log

Tracked issues found during development. Updated as we go.

| Date | Bug | Root Cause | Fix | Status |
|------|-----|------------|-----|--------|
| 2026-07-12 | NewsAPI returns `apiKeyInvalid` on live fetch | Code targeted **newsapi.org** but key is from **newsapi.ai** (Event Registry) — different APIs | Migrated fetcher to `eventregistry.org/api/v1/article/getArticles` | fixed |
| 2026-07-12 | AP News + Reuters RSS return DNS errors in smoke test | Feed host lookup failed (`nodename nor servname provided`) — may be transient network issue or deprecated feed URLs | Re-test after network stable; swap feeds if still failing | open |

## Notes

- **Status values:** `open`, `fixed`, `wontfix`
- Add a row when we catch a bug in dev or testing — even if fixed immediately, the paper trail helps for interviews.
