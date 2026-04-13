"""QThread worker that automates Redfin browsing via JavaScript injection.

Pipeline order (image analysis is LAST — it's the most expensive step):
  1. Navigate to Redfin, search for location
  2. Apply preference filters (budget, beds, baths, sqft, property type)
  3. Scroll results and collect listing URLs
  4. Open each listing in a new tab, extract details + image URLs
  5. LAST: Run Gemma 4 (Ollama) image analysis on confirmed listings
"""

import hashlib
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from image_analyzer import analyze_listing_images


PREFS_PATH = Path(__file__).parent / "config" / "preferences.json"


def load_preferences() -> dict:
    with open(PREFS_PATH) as f:
        return json.load(f)


class AutomationWorker(QThread):
    """Drives the embedded QWebEngineView browser from a background thread.

    Communication with the main (UI) thread happens entirely through Qt signals.
    The worker emits *request* signals (navigate, run JS, open/close tabs) and
    blocks on threading.Events until the main thread fulfils them.
    """

    # -- Requests to the main thread --
    navigate_requested = Signal(str)              # url
    js_requested = Signal(str, int)               # js_code, request_id
    open_tab_requested = Signal(str, str)          # listing_id, url
    close_tab_requested = Signal(str)              # listing_id
    switch_to_main_tab = Signal()

    # -- Dashboard updates --
    status_update = Signal(str)
    log_message = Signal(str)
    listing_found = Signal(str)                    # JSON-encoded listing dict
    analysis_started = Signal(str)                 # listing_id
    analysis_complete = Signal(str, str)            # listing_id, JSON-encoded analysis
    agent_done = Signal(int, int)                   # total_found, total_analyzed
    agent_killed_signal = Signal()
    pause_toggled = Signal(bool)
    report_ready = Signal(str)                     # HTML report content

    def __init__(self):
        super().__init__()
        self.prefs = load_preferences()
        self.killed = False
        self.paused = False
        self._max_listings = 0  # 0 = no limit; set by main.py --test

        # Cross-thread synchronization for JS execution
        self._js_results: dict[int, object] = {}
        self._js_events: dict[int, threading.Event] = {}
        self._js_counter = 0

        # Navigation sync
        self._nav_event = threading.Event()

        # Tab sync
        self._tab_event = threading.Event()

        # Collected data
        self._listings: list[dict] = []
        self._seen_ids: set[str] = set()
        self._analyses: dict[str, dict] = {}  # listing_id -> analysis result

    # ------------------------------------------------------------------
    # Kill / pause (called from main thread)
    # ------------------------------------------------------------------
    def kill(self):
        self.killed = True

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_toggled.emit(self.paused)

    def _check(self):
        """Check kill/pause state. Call frequently from the work loop."""
        if self.killed:
            raise _KilledException()
        while self.paused:
            time.sleep(0.3)
            if self.killed:
                raise _KilledException()

    # ------------------------------------------------------------------
    # Cross-thread helpers
    # ------------------------------------------------------------------
    def _navigate(self, url: str):
        """Ask main thread to navigate, block until load finishes."""
        self._nav_event.clear()
        self.navigate_requested.emit(url)
        self._nav_event.wait(timeout=30)
        time.sleep(1)  # let JS settle

    def nav_finished(self):
        """Called by main thread when loadFinished fires."""
        self._nav_event.set()

    def _run_js(self, code: str, timeout: float = 10) -> object:
        """Ask main thread to execute JS, block until result arrives."""
        self._js_counter += 1
        rid = self._js_counter
        evt = threading.Event()
        self._js_events[rid] = evt
        self.js_requested.emit(code, rid)
        evt.wait(timeout=timeout)
        result = self._js_results.pop(rid, None)
        self._js_events.pop(rid, None)
        return result

    def js_result_ready(self, rid: int, value: object):
        """Called by main thread when runJavaScript returns."""
        self._js_results[rid] = value
        evt = self._js_events.get(rid)
        if evt:
            evt.set()

    def _open_tab(self, listing_id: str, url: str):
        """Ask main thread to open a new tab, block until loaded."""
        self._tab_event.clear()
        self.open_tab_requested.emit(listing_id, url)
        self._tab_event.wait(timeout=30)
        time.sleep(2)  # let page render

    def tab_loaded(self):
        """Called by main thread when the new tab finishes loading."""
        self._tab_event.set()

    def _close_tab(self, listing_id: str):
        self.close_tab_requested.emit(listing_id)
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # Main work loop
    # ------------------------------------------------------------------
    def run(self):
        try:
            self._browse_redfin()
        except _KilledException:
            self.agent_killed_signal.emit()
        except Exception as e:
            self.status_update.emit(f"Error: {e}")
            self.log_message.emit(f"Agent error: {e}")
            import traceback
            traceback.print_exc()

    def _browse_redfin(self):
        location = self.prefs["locations"][0]

        # Step 1: Navigate to Redfin
        self.status_update.emit("Navigating to Redfin...")
        self.log_message.emit("Opening redfin.com")
        self._navigate("https://www.redfin.com")
        time.sleep(2)
        self._check()

        # Step 2: Search for location
        self.status_update.emit(f"Searching for {location}...")
        self.log_message.emit(f"Searching: {location}")

        # Use React-compatible value setter for the search input
        escaped = location.replace("'", "\\'")
        self._run_js(f"""
            (function() {{
                var box = document.querySelector('#search-box-input');
                if (!box) return 'no_input';
                box.focus();
                var setter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value').set;
                setter.call(box, '{escaped}');
                box.dispatchEvent(new Event('input', {{bubbles: true}}));
                return 'ok';
            }})();
        """)
        time.sleep(2)
        self._check()

        # Click first autocomplete suggestion
        self._run_js("""
            (function() {
                var items = document.querySelectorAll(
                    '.SearchMenu .item-row, .searchInputNode .item-row, ' +
                    '[data-rf-test-id="search-menu"] .item-row'
                );
                if (items.length > 0) { items[0].click(); return 'clicked'; }
                // Fallback: submit search form
                var box = document.querySelector('#search-box-input');
                if (box) {
                    box.dispatchEvent(new KeyboardEvent('keydown',
                        {key:'Enter', code:'Enter', keyCode:13, bubbles:true}));
                    return 'enter';
                }
                return 'nothing';
            })();
        """)
        time.sleep(3)
        self._check()

        # Step 3: Apply filters via URL
        self.status_update.emit("Applying preference filters...")
        self.log_message.emit("Applying filters: price, beds, baths, sqft, type")
        self._apply_url_filters()
        self._check()

        # Step 4: Collect listings from results page
        self.status_update.emit("Scanning results page...")
        listing_stubs = self._collect_listings()
        self.log_message.emit(f"Found {len(listing_stubs)} listings on results page")
        self._check()

        if not listing_stubs:
            self.status_update.emit("No listings found matching filters")
            self.log_message.emit("No listings matched. Try adjusting preferences.")
            self.agent_done.emit(0, 0)
            return

        # Step 5: Open each listing, extract details + schools, filter
        # inline, post matches immediately, stop at --test limit.
        school_keywords = self._extract_school_keywords()
        matched = 0
        skipped = 0
        analysis_pool = ThreadPoolExecutor(max_workers=3)
        analysis_futures: list[Future] = []

        for i, stub in enumerate(listing_stubs):
            self._check()

            # Stop early if we've hit the --test limit
            if self._max_listings > 0 and matched >= self._max_listings:
                self.log_message.emit(
                    f"Test mode: reached {self._max_listings} matched listings, stopping"
                )
                break

            self.status_update.emit(
                f"Opening listing {i+1}/{len(listing_stubs)}: {stub.get('address', '?')}"
            )
            self.log_message.emit(f"Opening: {stub.get('address', stub['url'][:60])}")

            full_listing = self._extract_listing_details(stub)
            if not full_listing:
                continue

            # Apply school filter immediately
            if school_keywords:
                if not self._matches_school_requirement(full_listing, school_keywords):
                    skipped += 1
                    self.log_message.emit(
                        f"  SKIPPED — school mismatch "
                        f"(need: {', '.join(school_keywords)})"
                    )
                    continue

            # Listing passed all filters — post to dashboard, open tab
            matched += 1
            self._listings.append(full_listing)
            self.listing_found.emit(json.dumps(full_listing))
            self.log_message.emit(
                f"  MATCHED — opening in tab ({matched} found so far)"
            )
            self._open_tab(full_listing["id"], full_listing["url"])

            # Switch back to main tab so next listing extraction works
            self.switch_to_main_tab.emit()
            time.sleep(0.3)

            # Kick off image analysis in background thread
            lid = full_listing["id"]
            imgs = full_listing.get("image_urls", [])
            if imgs:
                self.analysis_started.emit(lid)
                future = analysis_pool.submit(
                    self._run_analysis_bg, full_listing
                )
                analysis_futures.append(future)

        if skipped:
            self.log_message.emit(f"School filter: skipped {skipped} listings")

        if not self._listings:
            self.status_update.emit("No listings passed filters")
            self.log_message.emit("No listings matched.")
            self.agent_done.emit(0, 0)
            analysis_pool.shutdown(wait=False)
            return

        # Wait for any still-running image analyses to finish
        self.status_update.emit(
            f"Waiting for {len(analysis_futures)} image analyses to complete..."
        )
        for future in analysis_futures:
            future.result()  # blocks until done

        analysis_pool.shutdown(wait=True)
        total_analyzed = len(self._analyses)

        # Generate final HTML report
        report_html = self._generate_report(self._listings, self._analyses)
        self.report_ready.emit(report_html)

        self.status_update.emit("Done")
        self.agent_done.emit(len(self._listings), total_analyzed)
        self.log_message.emit(
            f"Finished: {len(self._listings)} listings, {total_analyzed} analyzed"
        )

    # ------------------------------------------------------------------
    # Background image analysis
    # ------------------------------------------------------------------
    def _run_analysis_bg(self, listing: dict):
        """Run image analysis in a background thread. Emits signals on completion."""
        lid = listing["id"]
        imgs = listing.get("image_urls", [])
        addr = listing.get("address", "")
        self.log_message.emit(
            f"Analyzing images: {addr or '?'} ({len(imgs)} photos)"
        )
        try:
            analysis = analyze_listing_images(lid, imgs, address=addr)
            analysis["_image_urls"] = imgs  # pass through for pada overlay
            self._analyses[lid] = analysis
            self.analysis_complete.emit(lid, json.dumps(analysis))
            self.log_message.emit(
                f"Analysis done: {listing.get('address', '?')} — "
                f"score {analysis.get('condition_score', '?')}/10"
            )
        except Exception as e:
            self.log_message.emit(
                f"Analysis failed: {listing.get('address', '?')} — {e}"
            )

    # ------------------------------------------------------------------
    # URL-based filter application
    # ------------------------------------------------------------------
    def _apply_url_filters(self):
        # Read the current URL from the browser
        current_url = self._run_js("window.location.href") or ""

        parts = []
        budget = self.prefs.get("budget", {})
        if budget.get("min"):
            parts.append(f"min-price={budget['min']}")
        if budget.get("max"):
            parts.append(f"max-price={budget['max']}")
        if self.prefs.get("bedrooms_min"):
            parts.append(f"min-beds={self.prefs['bedrooms_min']}")
        if self.prefs.get("bathrooms_min"):
            parts.append(f"min-baths={self.prefs['bathrooms_min']}")
        if self.prefs.get("sqft_min"):
            parts.append(f"min-sqft={self.prefs['sqft_min']}-sqft")

        type_map = {
            "house": "house", "single-family": "house",
            "townhouse": "townhouse", "condo": "condo",
            "multi-family": "multifamily",
        }
        ptypes = self.prefs.get("property_types", [])
        mapped = [type_map.get(t, t) for t in ptypes]
        if mapped:
            parts.append(f"property-type={'+'.join(mapped)}")

        if not self.prefs.get("include_pending", True):
            parts.append("status=active")

        if not parts:
            return

        filter_segment = ",".join(parts)
        base = re.sub(r"/filter/.*$", "", current_url.rstrip("/"))
        filtered_url = f"{base}/filter/{filter_segment}"

        self._navigate(filtered_url)
        time.sleep(2)
        self.log_message.emit(f"Filters applied via URL")

    # ------------------------------------------------------------------
    # Collect listing stubs from the results page
    # ------------------------------------------------------------------
    def _collect_listings(self) -> list[dict]:
        # Scroll down to load more listings
        for _ in range(4):
            self._run_js("window.scrollBy(0, 800)")
            time.sleep(0.8)
        self._run_js("window.scrollTo(0, 0)")
        time.sleep(1)

        # Extract listing card data via JS
        raw = self._run_js("""
            (function() {
                var results = [];
                var seen = {};
                // Find all listing links
                var links = document.querySelectorAll(
                    'a[href*="/home/"], .HomeCardContainer a'
                );
                for (var i = 0; i < links.length; i++) {
                    var a = links[i];
                    var href = a.href || a.getAttribute('href') || '';
                    if (href.indexOf('/home/') === -1) continue;
                    if (href.startsWith('/')) href = 'https://www.redfin.com' + href;
                    if (seen[href]) continue;
                    seen[href] = true;

                    // Try to get address from the card
                    var card = a.closest('.HomeCardContainer') || a;
                    var addrEl = card.querySelector(
                        '.homeAddressV2, .bp-Homecard__Address, ' +
                        '[data-rf-test-id="homecard-street"]'
                    );
                    var addr = addrEl ? addrEl.innerText.trim() : '';

                    var priceEl = card.querySelector(
                        '.homecardV2Price span, .bp-Homecard__Price--value, ' +
                        '[data-rf-test-id="homecard-price"]'
                    );
                    var price = priceEl ? priceEl.innerText.trim() : '';

                    results.push({url: href, address: addr, price: price});
                }
                return JSON.stringify(results);
            })();
        """, timeout=15)

        if not raw:
            return []

        try:
            stubs = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

        # Assign IDs and deduplicate
        listings = []
        for stub in stubs:
            lid = hashlib.md5(stub["url"].encode()).hexdigest()[:12]
            if lid in self._seen_ids:
                continue
            self._seen_ids.add(lid)
            stub["id"] = lid
            if not stub["address"]:
                stub["address"] = (
                    stub["url"].split("/home/")[0].split("/")[-1]
                    .replace("-", " ").title()
                )
            listings.append(stub)

        return listings

    # ------------------------------------------------------------------
    # Open a listing in a new tab and extract full details + images
    # ------------------------------------------------------------------
    def _extract_listing_details(self, stub: dict) -> dict | None:
        try:
            self._open_tab(stub["id"], stub["url"])

            listing = {
                "id": stub["id"],
                "url": stub["url"],
                "address": stub.get("address", ""),
                "price": stub.get("price", ""),
                "beds": "",
                "baths": "",
                "sqft": "",
                "lot_size": "",
                "year_built": "",
                "schools": [],
                "image_urls": [],
            }

            # Extract property details via JS (runs on the listing tab)
            details_raw = self._run_js("""
                (function() {
                    function txt(sel) {
                        var el = document.querySelector(sel);
                        return el ? el.innerText.trim() : '';
                    }
                    return JSON.stringify({
                        address: txt('.street-address, .full-address, ' +
                                     '[data-rf-test-id="abp-streetLine"]'),
                        price: txt('[data-rf-test-id="abp-price"] .statsValue, ' +
                                   '.price-section .statsValue, .HomeInfoPrice .price'),
                        beds: txt('[data-rf-test-id="abp-beds"] .statsValue, ' +
                                  '.bed-icon + span'),
                        baths: txt('[data-rf-test-id="abp-baths"] .statsValue, ' +
                                   '.bath-icon + span'),
                        sqft: txt('[data-rf-test-id="abp-sqFt"] .statsValue, ' +
                                  '.sqft-icon + span'),
                        lot_size: txt('[data-rf-test-id="abp-lotSize"] .statsValue'),
                        year_built: txt('[data-rf-test-id="abp-yearBuilt"] .statsValue')
                    });
                })();
            """, timeout=10)

            if details_raw:
                try:
                    details = json.loads(details_raw)
                    for key, val in details.items():
                        if val:
                            listing[key] = val
                except (json.JSONDecodeError, TypeError):
                    pass

            # Scroll down slowly to trigger lazy-loaded sections (schools
            # section is far below the fold). Do NOT click anything — just
            # scroll so the browser renders the schools content.
            for scroll_pct in (25, 50, 75, 100):
                self._run_js(
                    f"window.scrollTo(0, document.body.scrollHeight * {scroll_pct}/100)"
                )
                time.sleep(0.6)
            time.sleep(1)

            # Extract ASSIGNED school names only.
            # Redfin layout (from screenshot):
            #   School Name (bold)
            #   Public K-5 • Assigned • 1.1mi
            #   [rating badge]
            #   ---
            #   Next School Name
            #   Public 6-8 • Assigned • 1.2mi
            #   ...
            #   "Show nearby schools ▼"   <-- DO NOT CLICK
            #
            # Strategy: grab the entire page text, find the Schools/Places/
            # Transit tab area, extract lines that are followed by a line
            # containing "Assigned".
            schools_raw = self._run_js("""
                (function() {
                    var assigned = [];
                    // Grab the full page text
                    var body = document.body.innerText;

                    // Find the schools section: starts around "Schools"
                    // tab and ends before "Show nearby" or next major section
                    var schoolsStart = body.indexOf('Schools');
                    if (schoolsStart === -1) return JSON.stringify([]);

                    // Cut to just the schools region
                    var region = body.substring(schoolsStart);

                    // Stop before "Show nearby schools" — everything after
                    // that is nearby, not assigned
                    var nearbyIdx = region.search(/Show\\s+nearby/i);
                    if (nearbyIdx > 0) {
                        region = region.substring(0, nearbyIdx);
                    }

                    // Also stop before "Provided by" footer
                    var providedIdx = region.indexOf('Provided by');
                    if (providedIdx > 0 && (nearbyIdx < 0 || providedIdx < nearbyIdx)) {
                        region = region.substring(0, providedIdx);
                    }

                    var lines = region.split('\\n')
                        .map(function(l) { return l.trim(); })
                        .filter(function(l) { return l.length > 0; });

                    // Walk lines: if the NEXT line contains "Assigned",
                    // the current line is the school name
                    for (var i = 0; i < lines.length - 1; i++) {
                        var nextLine = lines[i + 1];
                        if (nextLine.indexOf('Assigned') !== -1) {
                            var name = lines[i];
                            // Skip tab headers and noise
                            if (name === 'Schools') continue;
                            if (name === 'Places') continue;
                            if (name === 'Transit') continue;
                            if (name.length < 4) continue;
                            // Remove any trailing rating like "8/10 >"
                            name = name.replace(/\\d+\\/\\d+\\s*>?$/, '').trim();
                            if (name) assigned.push(name);
                        }
                    }
                    return JSON.stringify(assigned);
                })();
            """, timeout=10)

            if schools_raw:
                try:
                    listing["schools"] = json.loads(schools_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Log what we found for debugging
            school_names = listing.get("schools", [])
            if school_names:
                self.log_message.emit(
                    f"  Assigned schools: {', '.join(school_names)}"
                )
            else:
                self.log_message.emit(
                    f"  No assigned schools found on page"
                )

            # Scroll back to top for image extraction
            self._run_js("window.scrollTo(0, 0)")
            time.sleep(0.5)

            # Extract image URLs via JS — deduplicate by base photo ID
            # so we don't send the same image at multiple resolutions.
            # Only include actual property listing photos (not agent
            # headshots, brokerage logos, map tiles, or tiny icons).
            images_raw = self._run_js("""
                (function() {
                    var urls = [];
                    var seenUrl = {};
                    var seenBase = {};

                    // Normalize a Redfin image URL to a base key so
                    // different sizes of the same photo deduplicate.
                    function baseKey(url) {
                        return url
                            .replace(/\\/gen[A-Za-z]+\\d*\\./, '/gen.')
                            .replace(/_w\\d+(_h\\d+)?/g, '')
                            .replace(/\\/variant_w\\d+/g, '')
                            .replace(/\\?.*$/, '');
                    }

                    // Patterns that indicate non-property images
                    var excludePatterns = [
                        '/agent/', '/agents/', '/headshot', '/avatar',
                        '/logo', '/brokerage', '/office/',
                        'maps.googleapis.com', 'maps.google.com',
                        '/icon', '/badge', '/sprite',
                        'walk-score', 'walkscore',
                        '/static/', 'redfin.com/stingray/',
                        'redfin-agent', 'profilePhoto'
                    ];

                    function isExcluded(src) {
                        var lower = src.toLowerCase();
                        for (var k = 0; k < excludePatterns.length; k++) {
                            if (lower.indexOf(excludePatterns[k].toLowerCase()) !== -1) return true;
                        }
                        return false;
                    }

                    function addUrl(src, el) {
                        if (!src || seenUrl[src]) return;
                        if (src.indexOf('placeholder') !== -1) return;
                        if (src.indexOf('redfin') === -1 && src.indexOf('rdcpix') === -1) return;
                        if (isExcluded(src)) return;
                        // Skip tiny images (logos, icons, agent headshots)
                        if (el && el.naturalWidth > 0 && el.naturalWidth < 150) return;
                        if (el && el.naturalHeight > 0 && el.naturalHeight < 100) return;
                        seenUrl[src] = true;
                        var bk = baseKey(src);
                        if (seenBase[bk]) return;
                        seenBase[bk] = true;
                        urls.push(src);
                    }

                    // Method 1: img tags — target photo gallery containers
                    var imgs = document.querySelectorAll(
                        '.InlinePhotoPreview img, .bp-MediaShowcase img, ' +
                        '.HomeViewPhotos img, [data-rf-test-id="photo-container"] img, ' +
                        '.PhotosView img, .listingPhoto img, ' +
                        'img[src*="ssl.cdn-redfin.com/photo"], img[src*="rdcpix"]'
                    );
                    for (var i = 0; i < imgs.length; i++) {
                        addUrl(imgs[i].src || '', imgs[i]);
                    }
                    // Method 2: background-image styles in photo containers
                    var bgs = document.querySelectorAll(
                        '.InlinePhotoPreview [style*="background-image"], ' +
                        '.bp-MediaShowcase [style*="background-image"], ' +
                        '.HomeViewPhotos [style*="background-image"], ' +
                        '.PhotosView [style*="background-image"], ' +
                        '[data-rf-test-id="photo-container"] [style*="background-image"]'
                    );
                    for (var j = 0; j < bgs.length; j++) {
                        var style = bgs[j].getAttribute('style') || '';
                        var m = style.match(/url\\(['"]?(https?:\\/\\/[^'"\\)]+)['"]?\\)/);
                        if (m && m[1] && !isExcluded(m[1])) addUrl(m[1], null);
                    }
                    return JSON.stringify(urls);
                })();
            """, timeout=10)

            if images_raw:
                try:
                    listing["image_urls"] = json.loads(images_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Close the listing tab
            self._close_tab(stub["id"])
            return listing

        except _KilledException:
            raise
        except Exception as e:
            print(f"  [automation] Error extracting listing: {e}")
            try:
                self._close_tab(stub["id"])
            except Exception:
                pass
            return None


    # ------------------------------------------------------------------
    # School / must-haves filtering
    # ------------------------------------------------------------------
    def _extract_school_keywords(self) -> list[str]:
        """Pull school name keywords from must_haves and neighborhood_notes.

        Looks for entries mentioning "school" and extracts the school names.
        Returns lowercased keyword fragments like ["college wood", "west clay"].
        """
        import re as _re
        keywords = []
        sources = (
            self.prefs.get("must_haves", []) +
            [self.prefs.get("neighborhood_notes", "")]
        )
        for entry in sources:
            if not entry:
                continue
            low = entry.lower()
            if "school" not in low and "elementary" not in low:
                continue

            # Match patterns like:
            #   "College Wood Elementary"
            #   "College Wood Elemntary"  (typo-tolerant)
            #   "West Clay Elementary school"
            # Capture: one or more capitalized words before a school-type word
            names = _re.findall(
                r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+'
                r'(?:[Ee]lement\w*|[Mm]iddle|[Hh]igh|[Ss]chool)',
                entry,
            )
            for name in names:
                kw = name.strip().lower()
                # Filter out noise words that aren't school names
                if kw in ("the", "must", "be", "or", "elementary"):
                    continue
                if len(kw) < 3:
                    continue
                keywords.append(kw)

        # Deduplicate
        result = list(dict.fromkeys(keywords))
        if result:
            self.log_message.emit(f"School keywords from preferences: {result}")
        return result

    @staticmethod
    def _matches_school_requirement(
        listing: dict, school_keywords: list[str]
    ) -> bool:
        """Return True if ANY of the listing's assigned schools contain
        ANY of the required school keywords.

        If no school data was extracted, the listing is REJECTED — we
        cannot confirm it meets the school requirement.
        """
        schools = listing.get("schools", [])
        if not schools:
            # No school data found — reject since we can't confirm
            return False
        for school_name in schools:
            school_lower = school_name.lower()
            for keyword in school_keywords:
                if keyword in school_lower:
                    return True
        return False


    # ------------------------------------------------------------------
    # HTML report generation
    # ------------------------------------------------------------------
    def _generate_report(
        self, listings: list[dict], analyses: dict[str, dict]
    ) -> str:
        """Build a self-contained HTML report with links to listing tabs."""
        rows = []
        for listing in listings:
            lid = listing["id"]
            a = analyses.get(lid, {})
            score = a.get("condition_score", "—")
            vastu = a.get("vastu_score", "—")
            feng = a.get("feng_shui_score", "—")
            summary = a.get("summary", "")
            maint = a.get("estimated_maintenance", {})
            maint_str = ""
            if maint.get("low") is not None:
                maint_str = f"${maint['low']:,} – ${maint.get('high', 0):,}"

            buyer_cost = a.get("estimated_buyer_cost", {})
            buyer_str = ""
            if buyer_cost and buyer_cost.get("low") is not None:
                buyer_str = f"${buyer_cost['low']:,} – ${buyer_cost.get('high', 0):,}"

            neg_price = a.get("negotiation_price", {})
            neg_str = ""
            if neg_price and neg_price.get("low") is not None:
                neg_str = f"${neg_price['low']:,} – ${neg_price.get('high', 0):,}"

            findings_html = ""
            for f in a.get("findings", []):
                sev = f.get("severity", "INFO")
                cat = f.get("category", "")
                desc = f.get("description", "")
                action = f.get("action", "")
                findings_html += (
                    f'<tr>'
                    f'<td><span class="sev {sev}">{sev}</span></td>'
                    f'<td class="cat">{cat}</td>'
                    f'<td>{desc}'
                    f'{"<br><em>" + action + "</em>" if action else ""}'
                    f'</td></tr>'
                )

            remedies_html = ""
            for r in a.get("vastu_remedies", []):
                remedies_html += f"<li>{r}</li>"
            cures_html = ""
            for c in a.get("feng_shui_cures", []):
                cures_html += f"<li>{c}</li>"

            priorities_html = ""
            for p in a.get("inspection_priorities", []):
                priorities_html += f"<li>{p}</li>"

            pada = a.get("pada_analysis")
            pada_html = ""
            if pada and pada.get("is_south_facing"):
                # Reverse padas for display: SW (left) → SE (right)
                all_padas = pada.get("all_padas", [])
                reversed_padas = list(reversed(all_padas))

                # Build entrance image with overlay
                img_urls = listing.get("image_urls", [])
                img_idx = pada.get("entrance_image_index", 0) or 0
                entrance_img_html = ""
                if img_idx < len(img_urls):
                    overlay_cells = ""
                    for p in reversed_padas:
                        oc = "good" if p.get("auspicious") else "bad"
                        if p.get("has_door"):
                            oc += " door"
                        overlay_cells += (
                            f'<div class="pada-overlay-cell {oc}">'
                            f'<span class="pada-num">{p["number"]}</span>'
                            f'<span class="pada-name">{p["name"]}</span>'
                            f'</div>'
                        )
                    entrance_img_html = (
                        f'<div class="pada-image-wrap">'
                        f'<img src="{img_urls[img_idx]}" class="pada-entrance-img">'
                        f'<div class="pada-overlay-grid">{overlay_cells}</div>'
                        f'<div class="pada-dir-labels">'
                        f'<span class="pada-dir-label left">SW</span>'
                        f'<span class="pada-dir-label right">SE</span>'
                        f'</div>'
                        f'</div>'
                    )

                pada_cells = ""
                for p in reversed_padas:
                    cell_cls = "good" if p.get("auspicious") else "bad"
                    if p.get("has_door"):
                        cell_cls += " door"
                    pada_cells += (
                        f'<div class="pada-cell {cell_cls}">'
                        f'<span class="pada-num">{p["number"]}</span>'
                        f'<span class="pada-name">{p["name"]}</span>'
                        f'</div>'
                    )
                result_cls = "auspicious" if pada.get("is_auspicious") else "inauspicious"
                result_icon = "\u2714" if pada.get("is_auspicious") else "\u2718"
                pada_html = (
                    f'<div class="pada-section">'
                    f'<div class="pada-title">South-Facing Vastu Pada Analysis</div>'
                    f'{entrance_img_html}'
                    f'<div style="font-size:10px;color:var(--muted);margin-bottom:4px;">SW \u2190 Padas 9\u20131 \u2192 SE</div>'
                    f'<div class="pada-grid">{pada_cells}</div>'
                    f'<div class="pada-result {result_cls}">'
                    f'{result_icon} Door in Pada {pada.get("estimated_pada", "?")} '
                    f'({pada.get("pada_name", "?")}) \u2014 '
                    f'{"Auspicious" if pada.get("is_auspicious") else "Inauspicious"}'
                    f': {pada.get("effect", "")}'
                    f'</div>'
                    f'<div class="pada-confidence">Confidence: {pada.get("confidence", "N/A")}'
                    f'{" \u2014 " + pada["notes"] if pada.get("notes") else ""}</div>'
                    f'</div>'
                )

            thumbs = "".join(
                f'<img src="{u}" loading="lazy">'
                for u in listing.get("image_urls", [])[:6]
            )

            score_cls = "good" if isinstance(score, (int, float)) and score >= 7 else "fair" if isinstance(score, (int, float)) and score >= 5 else "poor"
            vastu_cls = "good" if isinstance(vastu, (int, float)) and vastu >= 7 else "fair" if isinstance(vastu, (int, float)) and vastu >= 5 else "poor"
            feng_cls = "good" if isinstance(feng, (int, float)) and feng >= 7 else "fair" if isinstance(feng, (int, float)) and feng >= 5 else "poor"

            rows.append(f"""
            <div class="listing" id="report-{lid}">
              <div class="listing-header">
                <div>
                  <a href="#" class="listing-link" onclick="switchToTab('{lid}'); return false;">
                    {listing.get("address", "Address unavailable")}
                  </a>
                  <div class="listing-meta">
                    {listing.get("beds", "")} {"beds" if listing.get("beds") else ""} &middot;
                    {listing.get("baths", "")} {"baths" if listing.get("baths") else ""} &middot;
                    {listing.get("sqft", "")} {"sqft" if listing.get("sqft") else ""}
                    {" &middot; Lot: " + listing.get("lot_size") if listing.get("lot_size") else ""}
                    {" &middot; Built " + listing.get("year_built") if listing.get("year_built") else ""}
                  </div>
                  {f'<div class="schools">Schools: {", ".join(listing.get("schools", []))}</div>' if listing.get("schools") else ""}
                </div>
                <div class="listing-price">{listing.get("price", "")}</div>
              </div>
              {f'<div class="thumb-row">{thumbs}</div>' if thumbs else ""}
              <div class="scores">
                <span class="badge {score_cls}">Condition: {score}/10</span>
                {f'<span class="badge {vastu_cls}">Vastu: {vastu}/10</span>' if vastu != "—" else ""}
                {f'<span class="badge {feng_cls}">Feng Shui: {feng}/10</span>' if feng != "—" else ""}
              </div>
              <p class="summary">{summary}</p>
              {f'<table class="findings"><tr><th>Severity</th><th>Category</th><th>Finding</th></tr>{findings_html}</table>' if findings_html else ""}
              {f'<div class="maint">Est. Maintenance: <strong>{maint_str}</strong></div>' if maint_str else ""}
              {f'<div class="maint">Est. Total Buyer Cost: <strong>{buyer_str}</strong><br><span style="font-size:10px;color:var(--muted);">{buyer_cost.get("notes", "")}</span></div>' if buyer_str else ""}
              {f'<div class="maint" style="border-left:3px solid var(--green);">Negotiate At: <strong style="color:var(--green);">{neg_str}</strong><br><span style="font-size:10px;color:var(--muted);">{neg_price.get("reasoning", "")}</span></div>' if neg_str else ""}
              {f'<div class="remedies"><strong>Vastu Remedies:</strong><ul>{remedies_html}</ul></div>' if remedies_html else ""}
              {f'<div class="cures"><strong>Feng Shui Cures:</strong><ul>{cures_html}</ul></div>' if cures_html else ""}
              {pada_html}
              {f'<div class="priorities"><strong>Inspection Priorities:</strong><ul>{priorities_html}</ul></div>' if priorities_html else ""}
            </div>
            """)

        listing_rows = "\n".join(rows) if rows else '<p class="empty">No listings analyzed.</p>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Analysis Report</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3345; --text: #e4e6ef; --muted: #8b8fa3;
    --accent: #6c8cff; --green: #34d399; --yellow: #fbbf24;
    --orange: #fb923c; --red: #f87171;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 20px;
  }}
  h1 {{ font-size: 20px; margin-bottom: 6px; }}
  h1 span {{ color: var(--accent); }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
  .listing {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 16px; overflow: hidden;
  }}
  .listing-header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 14px 18px; background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }}
  .listing-link {{
    font-size: 15px; font-weight: 600; color: var(--accent);
    text-decoration: none; cursor: pointer;
  }}
  .listing-link:hover {{ text-decoration: underline; }}
  .listing-meta {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .schools {{ font-size: 12px; color: var(--green); margin-top: 3px; }}
  .listing-price {{ font-size: 17px; font-weight: 700; color: var(--green); }}
  .thumb-row {{
    display: flex; gap: 3px; padding: 8px 18px; overflow-x: auto;
  }}
  .thumb-row img {{ height: 65px; border-radius: 4px; object-fit: cover; }}
  .scores {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 10px 18px; }}
  .badge {{
    padding: 3px 10px; border-radius: 16px; font-size: 13px; font-weight: 600;
  }}
  .badge.good {{ background: rgba(52,211,153,.15); color: var(--green); }}
  .badge.fair {{ background: rgba(251,191,36,.15); color: var(--yellow); }}
  .badge.poor {{ background: rgba(248,113,113,.15); color: var(--red); }}
  .summary {{ padding: 8px 18px; font-size: 13px; line-height: 1.5; }}
  .findings {{
    width: 100%; border-collapse: collapse; margin: 0;
    font-size: 12px;
  }}
  .findings th {{
    text-align: left; padding: 6px 18px; background: var(--surface2);
    color: var(--muted); font-size: 11px; text-transform: uppercase;
  }}
  .findings td {{ padding: 6px 18px; border-top: 1px solid var(--border); }}
  .findings em {{ color: var(--muted); font-size: 11px; }}
  .cat {{ color: var(--muted); font-size: 11px; }}
  .sev {{
    font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 3px;
    text-transform: uppercase;
  }}
  .sev.INFO     {{ background: rgba(108,140,255,.15); color: var(--accent); }}
  .sev.MINOR    {{ background: rgba(52,211,153,.15);  color: var(--green); }}
  .sev.MODERATE {{ background: rgba(251,191,36,.15);  color: var(--yellow); }}
  .sev.MAJOR    {{ background: rgba(251,146,60,.15);  color: var(--orange); }}
  .sev.CRITICAL {{ background: rgba(239,68,68,.15);   color: var(--red); }}
  .maint {{
    padding: 8px 18px; font-size: 12px; background: var(--surface2);
    margin: 8px 18px; border-radius: 6px;
  }}
  .maint strong {{ color: var(--accent); }}
  .remedies, .cures, .priorities {{
    padding: 6px 18px; font-size: 12px; color: var(--muted);
  }}
  .remedies strong {{ color: var(--yellow); }}
  .cures strong {{ color: var(--green); }}
  .priorities strong {{ color: var(--text); }}
  ul {{ margin: 4px 0 4px 20px; }}
  li {{ margin: 2px 0; }}
  .pada-section {{ margin: 8px 18px; padding: 8px 12px; background: var(--surface2); border-radius: 6px; }}
  .pada-title {{ font-size: 12px; font-weight: 600; margin-bottom: 6px; }}
  .pada-grid {{ display: grid; grid-template-columns: repeat(9, 1fr); gap: 2px; margin-bottom: 6px; }}
  .pada-cell {{
    text-align: center; padding: 6px 2px; border-radius: 4px; font-size: 9px;
    border: 1px solid var(--border);
  }}
  .pada-cell .pada-num {{ font-weight: 700; font-size: 11px; }}
  .pada-cell .pada-name {{ display: block; margin-top: 1px; }}
  .pada-cell.good {{ background: rgba(52,211,153,.12); border-color: var(--green); }}
  .pada-cell.bad  {{ background: rgba(248,113,113,.10); border-color: var(--red); }}
  .pada-cell.door {{ outline: 2px solid var(--accent); outline-offset: -1px; }}
  .pada-confidence {{ font-size: 10px; color: var(--muted); margin-top: 4px; }}
  .pada-result {{ font-size: 12px; margin-top: 4px; }}
  .pada-result.auspicious {{ color: var(--green); }}
  .pada-result.inauspicious {{ color: var(--red); }}
  .pada-image-wrap {{
    position: relative; margin-bottom: 8px; border-radius: 6px; overflow: hidden;
  }}
  .pada-entrance-img {{ width: 100%; display: block; border-radius: 6px; }}
  .pada-overlay-grid {{
    position: absolute; bottom: 0; left: 0; right: 0; height: 35%;
    display: grid; grid-template-columns: repeat(9, 1fr); gap: 1px;
  }}
  .pada-overlay-cell {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    font-size: 9px; color: #fff; text-shadow: 0 1px 3px rgba(0,0,0,.8);
  }}
  .pada-overlay-cell.good {{ background: rgba(52,211,153,.35); }}
  .pada-overlay-cell.bad  {{ background: rgba(248,113,113,.30); }}
  .pada-overlay-cell.door {{
    outline: 2px solid var(--accent); outline-offset: -1px;
    background: rgba(108,140,255,.40);
  }}
  .pada-overlay-cell .pada-num {{ font-weight: 700; font-size: 12px; }}
  .pada-overlay-cell .pada-name {{ font-size: 8px; }}
  .pada-dir-labels {{
    position: absolute; bottom: 2px; left: 0; right: 0;
    display: flex; justify-content: space-between; padding: 0 4px;
    pointer-events: none;
  }}
  .pada-dir-label {{
    font-size: 10px; font-weight: 700; color: #fff;
    text-shadow: 0 1px 4px rgba(0,0,0,.9);
    background: rgba(0,0,0,.4); padding: 1px 5px; border-radius: 3px;
  }}
  .empty {{ text-align: center; color: var(--muted); padding: 40px; }}
</style>
</head>
<body>
<h1><span>RE Agent</span> v2 — Analysis Report</h1>
<div class="subtitle">
  {len(listings)} listings analyzed &middot;
  Click an address to jump to its browser tab
</div>
{listing_rows}

<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
var reportBridge = null;
new QWebChannel(qt.webChannelTransport, function(channel) {{
  reportBridge = channel.objects.reportBridge;
}});
function switchToTab(listingId) {{
  if (reportBridge) reportBridge.switchToTab(listingId);
}}
</script>
</body>
</html>"""


class _KilledException(Exception):
    """Raised when the user hits the kill switch."""
