#!/usr/bin/env python3
"""Real Estate Scouting Agent v2 — Single-window embedded browser + dashboard.

Split-panel Qt application:
  LEFT  — Embedded Chromium browser (QWebEngineView) browsing Redfin
  RIGHT — Analysis dashboard with live status, listing cards, and Gemma 4 Vision results

Usage:
    uv run main.py              # Run the agent
    uv run main.py --test       # Test mode: limit to 3 listings before image analysis

Setup (first time):
    uv sync
    cp .env.example .env        # add your GOOGLE_MAPS_API_KEY
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from PySide6.QtCore import Qt, QUrl, QObject, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QTabWidget, QWidget, QVBoxLayout,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebChannel import QWebChannel

from browser_automation import AutomationWorker

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"


# ======================================================================
# Bridge object exposed to dashboard JS via QWebChannel
# ======================================================================
class DashboardBridge(QObject):
    """Python object exposed to dashboard JS as `bridge`."""

    def __init__(self, worker: AutomationWorker):
        super().__init__()
        self._worker = worker
        self._prefs_path = Path(__file__).parent / "config" / "preferences.json"

    @Slot()
    def kill(self):
        self._worker.kill()

    @Slot()
    def togglePause(self):
        self._worker.toggle_pause()

    @Slot(str)
    def savePrefs(self, prefs_json: str):
        """Save preferences JSON to config file and update the worker."""
        try:
            new_prefs = json.loads(prefs_json)
            # Preserve fields not in the form
            with open(self._prefs_path) as f:
                existing = json.load(f)
            existing.update(new_prefs)
            with open(self._prefs_path, "w") as f:
                json.dump(existing, f, indent=2)
                f.write("\n")
            self._worker.prefs = existing
        except Exception as e:
            print(f"[bridge] Failed to save prefs: {e}")


class ReportBridge(QObject):
    """Python object exposed to report JS as `reportBridge`."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self._window = window

    @Slot(str)
    def switchToTab(self, listing_id: str):
        view = self._window._tab_views.get(listing_id)
        if view:
            idx = self._window.browser_tabs.indexOf(view)
            if idx >= 0:
                self._window.browser_tabs.setCurrentIndex(idx)


# ======================================================================
# Main window
# ======================================================================
class MainWindow(QMainWindow):

    def __init__(self, max_listings: int = 0):
        super().__init__()
        self.setWindowTitle("Real Estate Agent v2")
        self.resize(1440, 900)
        self._max_listings = max_listings
        self._tab_views: dict[str, QWebEngineView] = {}  # listing_id -> view
        self._tab_indices: dict[str, int] = {}

        # --- Central splitter ---
        splitter = QSplitter(Qt.Horizontal)

        # --- LEFT: Browser with tabs ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.browser_tabs = QTabWidget()
        self.browser_tabs.setTabsClosable(True)
        self.browser_tabs.tabCloseRequested.connect(self._on_tab_close)
        left_layout.addWidget(self.browser_tabs)

        # Set a realistic User-Agent on the default profile
        profile = QWebEngineProfile.defaultProfile()
        profile.setHttpUserAgent(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )

        # Main browser tab
        self.main_view = QWebEngineView()
        self.browser_tabs.addTab(self.main_view, "Redfin")
        # Prevent closing the main tab
        from PySide6.QtWidgets import QTabBar
        self.browser_tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)

        # --- RIGHT: Dashboard ---
        self.dashboard_view = QWebEngineView()
        dashboard_page = self.dashboard_view.page()

        # Set up QWebChannel for kill/pause bridge
        self._worker = AutomationWorker()
        if self._max_listings > 0:
            self._worker._max_listings = self._max_listings
        self._bridge = DashboardBridge(self._worker)
        channel = QWebChannel()
        channel.registerObject("bridge", self._bridge)
        dashboard_page.setWebChannel(channel)

        # Load dashboard HTML
        html_path = STATIC_DIR / "dashboard.html"
        with open(html_path) as f:
            html = f.read()
        self.dashboard_view.setHtml(html, QUrl("qrc:///"))

        # --- Assemble splitter ---
        splitter.addWidget(left)
        splitter.addWidget(self.dashboard_view)
        splitter.setSizes([780, 560])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        # --- Wire automation signals ---
        w = self._worker

        # Navigation: worker asks to navigate the main view
        w.navigate_requested.connect(self._on_navigate)
        w.js_requested.connect(self._on_run_js)
        w.open_tab_requested.connect(self._on_open_tab)
        w.close_tab_requested.connect(self._on_close_tab)
        w.switch_to_main_tab.connect(lambda: self.browser_tabs.setCurrentIndex(0))

        # Dashboard updates
        w.status_update.connect(self._dash_status)
        w.log_message.connect(self._dash_log)
        w.listing_found.connect(self._dash_listing)
        w.analysis_started.connect(self._dash_analysis_start)
        w.analysis_complete.connect(self._dash_analysis)
        w.agent_done.connect(self._dash_done)
        w.agent_killed_signal.connect(self._dash_killed)
        w.pause_toggled.connect(self._dash_pause_toggled)
        w.report_ready.connect(self._on_report_ready)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def start_agent(self):
        # Wait for the dashboard HTML to finish loading before starting
        # the worker — otherwise JS functions like setStatus() don't exist yet.
        def _on_dashboard_ready(ok):
            self.dashboard_view.loadFinished.disconnect(_on_dashboard_ready)
            # Load current preferences into the form
            prefs_json = json.dumps(self._worker.prefs)
            self._dash_js(f"loadPrefs({prefs_json})")
            self._worker.start()
        self.dashboard_view.loadFinished.connect(_on_dashboard_ready)

    def closeEvent(self, event):
        self._worker.kill()
        self._worker.wait(3000)
        event.accept()

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------
    def _on_tab_close(self, index):
        if index == 0:
            return  # don't close the main tab
        widget = self.browser_tabs.widget(index)
        self.browser_tabs.removeTab(index)
        # Remove from tracking
        for lid, view in list(self._tab_views.items()):
            if view is widget:
                del self._tab_views[lid]
                self._tab_indices.pop(lid, None)
                break
        widget.deleteLater()

    # ------------------------------------------------------------------
    # Signals from AutomationWorker
    # ------------------------------------------------------------------
    def _on_navigate(self, url: str):
        """Navigate the main browser tab to a URL."""
        def on_load(ok):
            self.main_view.loadFinished.disconnect(on_load)
            self._worker.nav_finished()
        self.main_view.loadFinished.connect(on_load)
        self.main_view.setUrl(QUrl(url))

    def _on_run_js(self, code: str, rid: int):
        """Execute JavaScript on the currently active browser tab."""
        # Determine which view to run JS on
        current_index = self.browser_tabs.currentIndex()
        if current_index == 0:
            page = self.main_view.page()
        else:
            widget = self.browser_tabs.widget(current_index)
            if isinstance(widget, QWebEngineView):
                page = widget.page()
            else:
                page = self.main_view.page()

        def callback(result):
            self._worker.js_result_ready(rid, result)
        page.runJavaScript(code, callback)

    def _on_open_tab(self, listing_id: str, url: str):
        """Open a listing in a new browser tab."""
        view = QWebEngineView()
        tab_index = self.browser_tabs.addTab(view, f"Loading...")
        self.browser_tabs.setCurrentIndex(tab_index)
        self._tab_views[listing_id] = view
        self._tab_indices[listing_id] = tab_index

        def on_load(ok):
            view.loadFinished.disconnect(on_load)
            # Update tab title with a short address
            title = view.title() or url.split("/")[-2].replace("-", " ")[:25]
            self.browser_tabs.setTabText(
                self.browser_tabs.indexOf(view), title[:25]
            )
            self._worker.tab_loaded()
        view.loadFinished.connect(on_load)
        view.setUrl(QUrl(url))

    def _on_close_tab(self, listing_id: str):
        """Close a listing tab."""
        view = self._tab_views.pop(listing_id, None)
        self._tab_indices.pop(listing_id, None)
        if view:
            idx = self.browser_tabs.indexOf(view)
            if idx >= 0:
                self.browser_tabs.removeTab(idx)
            view.deleteLater()
        # Switch back to main tab
        self.browser_tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Dashboard JS calls (right panel updates)
    # ------------------------------------------------------------------
    def _dash_js(self, js: str):
        self.dashboard_view.page().runJavaScript(js)

    def _dash_status(self, msg: str):
        self._dash_js(f"setStatus({json.dumps(msg)})")

    def _dash_log(self, msg: str):
        self._dash_js(f"addLog({json.dumps(msg)})")

    def _dash_listing(self, listing_json: str):
        self._dash_js(f"addListing({listing_json})")

    def _dash_analysis_start(self, listing_id: str):
        self._dash_js(f"startAnalysis({json.dumps(listing_id)})")

    def _dash_analysis(self, listing_id: str, analysis_json: str):
        self._dash_js(f"updateAnalysis({json.dumps(listing_id)}, {analysis_json})")

    def _dash_done(self, total_found: int, total_analyzed: int):
        self._dash_js(f"markDone({total_found}, {total_analyzed})")

    def _dash_killed(self):
        self._dash_js("markKilled()")

    def _dash_pause_toggled(self, is_paused: bool):
        self._dash_js(f"setPaused({'true' if is_paused else 'false'})")

    # ------------------------------------------------------------------
    # Report tab
    # ------------------------------------------------------------------
    def _on_report_ready(self, html: str):
        """Open the final analysis report in a new browser tab."""
        report_view = QWebEngineView()
        report_page = report_view.page()

        # Set up QWebChannel so report links can switch listing tabs
        report_bridge = ReportBridge(self)
        report_channel = QWebChannel()
        report_channel.registerObject("reportBridge", report_bridge)
        report_page.setWebChannel(report_channel)

        # Keep references alive
        self._report_bridge = report_bridge
        self._report_channel = report_channel

        report_view.setHtml(html, QUrl("qrc:///"))
        tab_index = self.browser_tabs.addTab(report_view, "Report")
        self.browser_tabs.setCurrentIndex(tab_index)


# ======================================================================
# Entry point
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Real Estate Agent v2")
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: limit to 3 listings before running image analysis",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Real Estate Agent v2")

    max_listings = 1 if args.test else 0
    window = MainWindow(max_listings=max_listings)
    window.show()
    window.start_agent()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
