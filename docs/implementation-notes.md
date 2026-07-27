# Implementation Notes: Design Decisions, Logic, Understanding

## Phase 1: Fetching News Articles
File-by-file walkthrough of code and design decisions

### db/schema.sql
- two tables are defined: ```articles``` and ```digests```
- URL is set to be unique in the data and title is not because articles should not be duplicated, even though their titles may be the same (may cover different content)
- id is SERIAL; SERIAL is a pseudo-type that tells PostgresSQL to execute multiple commands behind the scenes
- ```sentiment_score```/ ```is_duplicate``` / ```summary```
- articles.url is unique because there is at most one row per article link
- digests.date is unique because there is at most one digest per calendar day

### app/config.py
- newsapi_key is not required because RSS can be used to fetch
- RSS is Really Simple Syndication, standardized web feed format that auto-delivers updates from websites, blogs, and news sources
- @lrucache caches the first Settings so that env doesn't need to be re-read ina ny call; resets for testing and env changes

### db/connection.py
- Context managers ensure that there are no leaked connections
- Commit means persisting the action, and rollback is for undoing changes after a DB error
- Difference for connection-per-request and connection pool: how connection pools are managed and reused
    - connection pooling is better practice because it is lower latency and more scalable; the connection is opened once and kept open

### db/articles.py
- articles.py adds articles to the database, adding (title, url, source, published_at, full_text) for each article
- SQL ```ON CONFLICT ``` is used to execute a specific action when there is a conflict with the intended behavior, such as adding an item that has a value that already exists in the table when it is supposed to be unique
- ```RETURNING``` is a feature native to PostgreSQL in which the connection returns a particular value rather than a separate ```SELECT``` having to be processed
- Dataclass is used here because this is internal info and lightweight

### app/fetcher.py
- fetcher.py fetches the articles from News API and RSS feeds
- UTC is used instead of just date in order ot account for local time zones
- helper functions are defined at the top


### app/schemas.py
- Pydantic is used to assert more control over the inputs as we do not know what is in them

### app/main.py