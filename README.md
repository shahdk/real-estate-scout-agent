# Real Estate Agent v2 — Architecture

A desktop application that scouts Redfin listings against user preferences, then runs a local vision LLM (Gemma via Ollama) to inspect property photos for condition issues plus Vastu / Feng Shui assessment. Everything runs in a single Qt window: the embedded Chromium browser drives Redfin on the left, a live HTML dashboard streams results on the right.

## Stack

- **Python 3.11–3.13**, dependency-managed by `uv` (`pyproject.toml`, `uv.lock`)
- **PySide6 / Qt WebEngine** — the embedded Chromium browser, tab widget, and `QWebChannel` bridge to dashboard JS
- **httpx** — image fetches and Ollama HTTP calls
- **python-dotenv** — loads `GOOGLE_MAPS_API_KEY` and Ollama overrides from `.env`
- **Ollama (local)** — serves `gemma4:31b` for multimodal image analysis
- **Google Static Maps API** — satellite tile used to determine the property's facing direction

## Top-level layout

```
real-estate-agent-v2/
├── main.py                # Qt window, tabs, dashboard bridge, signal wiring
├── browser_automation.py  # AutomationWorker QThread — drives Redfin via JS injection
├── image_analyzer.py      # Gemma/Ollama vision pipeline + on-disk cache
├── static/dashboard.html  # Right-panel dashboard UI (single-file HTML/CSS/JS)
├── config/preferences.json# User search criteria (locations, budget, beds, schools, …)
├── cache/                 # Persisted analysis cache (analysis_cache.json)
├── .env                   # GOOGLE_MAPS_API_KEY, optional OLLAMA_BASE_URL/MODEL
└── pyproject.toml
```

## The three modules

### `main.py` — UI shell and signal router
- `MainWindow` builds a `QSplitter` with a `QTabWidget` of `QWebEngineView`s on the left and a single dashboard `QWebEngineView` on the right.
- Owns the `AutomationWorker` thread and connects every worker signal to a UI slot. The UI thread is the *only* thread allowed to touch `QWebEngineView`.
- `DashboardBridge` is registered into a `QWebChannel` so dashboard JS can call back into Python: `kill()`, `togglePause()`, `savePrefs(json)`.
- `ReportBridge` does the same for the final HTML report so "open listing" links can switch tabs.
- Slots fulfil worker requests:
  - `_on_navigate` → `view.setUrl(...)`, then `worker.nav_finished()` once `loadFinished` fires.
  - `_on_run_js` → `page.runJavaScript(code, callback)`, callback returns the result via `worker.js_result_ready(rid, value)`.
  - `_on_open_tab` / `_on_close_tab` manage per-listing tabs and notify the worker when load finishes.
- Dashboard updates are pushed by serializing the worker's signal payloads to JSON and calling JS functions inside `dashboard.html` (`setStatus`, `addListing`, `updateAnalysis`, `markDone`, …).

### `browser_automation.py` — `AutomationWorker(QThread)`
The brain. Runs entirely in a background QThread and never touches Qt widgets directly — it talks to the UI exclusively through Qt signals and `threading.Event`s.

**Pipeline (`_browse_redfin`)**
1. Navigate to `redfin.com`.
2. Type the location into `#search-box-input` using a React-compatible value setter, then click the first autocomplete row (or fall back to Enter).
3. Apply filters by rewriting the URL into `/filter/min-price=…,max-price=…,min-beds=…,…` — much more reliable than clicking the filter UI.
4. Scroll the results page, then `document.querySelectorAll('a[href*="/home/"]')` to collect listing stubs (URL, address, price). Each gets a stable `id = md5(url)[:12]`.
5. For each stub: open in a new tab, extract full details + image URLs from the listing page, apply the school-name filter from `preferences.json`, and emit `listing_found` so the dashboard renders a card.
6. **As soon as a listing matches**, the photos are submitted to a `ThreadPoolExecutor(max_workers=3)` running `_run_analysis_bg → analyze_listing_images`. Image analysis runs in parallel with continued browsing — it's the slowest step so it's pipelined, not blocking.
7. After all listings are processed, the worker waits on the analysis futures, builds an HTML report from `_listings + _analyses`, and emits `report_ready` so `MainWindow` opens it as a new tab.

**Cross-thread plumbing**
- `_run_js(code)` increments a request id, registers a `threading.Event`, emits `js_requested`, and blocks. The UI's `runJavaScript` callback calls `js_result_ready(rid, value)` which sets the event. Same pattern for `_navigate` (`_nav_event`) and `_open_tab` (`_tab_event`).
- `_check()` is called between every step and inside loops; it raises `_KilledException` if `kill()` was clicked, or busy-waits while `paused` is `True`.

**Signals out**
- *Browser commands:* `navigate_requested`, `js_requested`, `open_tab_requested`, `close_tab_requested`, `switch_to_main_tab`
- *Dashboard updates:* `status_update`, `log_message`, `listing_found`, `analysis_started`, `analysis_complete`, `agent_done`, `agent_killed_signal`, `pause_toggled`, `report_ready`

### `image_analyzer.py` — Gemma vision pipeline
- `analyze_listing_images(listing_id, image_urls, address)` is the entry point.
- **Cache-first**: `cache/analysis_cache.json` is read once (lazy, lock-protected). If `listing_id` is present, the cached dict is returned and Ollama is never called.
- Otherwise: download every image with `httpx`, base64-encode it, drop tiny / SVG / GIF responses, and append it to the prompt.
- If `GOOGLE_MAPS_API_KEY` is set, `fetch_satellite_image(address)` pulls a Google Static Maps satellite tile (zoom 20, red marker on the address) and appends it as the **last** image. The prompt instructs the model to use that tile (north = top) to determine the house's facing direction — the prerequisite for the Vastu pada analysis.
- The prompt (`CONDITION_PROMPT`) covers four parts: condition inspection, Vastu Shastra, Feng Shui, and a conditional "South-facing pada" analysis. The model is required to return strict JSON with `condition_score`, `vastu_score`, `feng_shui_score`, `findings[]`, `estimated_maintenance`, `estimated_buyer_cost`, `negotiation_price`, `pada_analysis`, etc.
- The result is written back into the cache and returned. The worker stashes it in `self._analyses[lid]` and emits `analysis_complete`.

## Configuration

- **`config/preferences.json`** — single source of truth for search criteria: `locations`, `budget {min,max}`, `property_types`, `bedrooms_min`, `bathrooms_min`, `sqft_min`, `must_haves`, `deal_breakers`, `neighborhood_notes` (the school-name filter is parsed out of this), `include_pending`. The dashboard's prefs form posts edits back via `bridge.savePrefs(json)` which merges into the file and updates `worker.prefs` in place.
- **`.env`** — `GOOGLE_MAPS_API_KEY` (required for satellite/pada analysis), optional `OLLAMA_BASE_URL` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default `gemma4:31b`).

## Data flow at a glance

```
preferences.json ──► AutomationWorker.prefs
                         │
                         ▼
           ┌─── JS injection ──► QWebEngineView (Redfin)
           │                          │
           │      stubs + details ◄───┘
           │
           ├─► listing_found signal ─────► dashboard.html (listing card)
           │
           └─► ThreadPoolExecutor ──► image_analyzer
                                          │
                                          ├── httpx → Redfin photos
                                          ├── httpx → Google Static Maps
                                          ├── httpx → Ollama (gemma4:31b)
                                          └── cache/analysis_cache.json
                                                        │
                       analysis_complete signal ◄───────┘
                                │
                                ▼
                          dashboard.html (score, findings, costs)
                                │
                       agent_done → report_ready
                                │
                                ▼
                       Final HTML report tab
```

## Threading model

- **UI thread (Qt main):** owns every `QWebEngineView`, runs JS, paints the dashboard.
- **Worker thread (`AutomationWorker`, QThread):** runs the entire scraping pipeline. Never touches widgets — only emits signals and waits on `threading.Event`s.
- **Analysis pool (`ThreadPoolExecutor`, max_workers=3):** runs `analyze_listing_images` so multiple Ollama calls overlap with continued browsing. Each task is independent and only writes to `self._analyses[lid]` (one key per task) and the analyzer's lock-protected cache.

This split is what makes "embedded browser + live dashboard + slow vision model" feel responsive: the user can pause/kill at any moment, browse tabs manually, or edit preferences while listings are still streaming in.

## Running

```bash
uv sync
cp .env.example .env        # add GOOGLE_MAPS_API_KEY
ollama pull gemma4:31b      # or override OLLAMA_MODEL
uv run main.py              # full run
uv run main.py --test       # stop after the first matched listing
```
