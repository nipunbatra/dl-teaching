#!/usr/bin/env python3
"""Serve source-aware Typst slide review pages with warm incremental rebuilds.

The long-running ``typst watch`` process keeps the compiler warm. When the main
deck source changes, this launcher identifies the affected physical slide(s),
waits for the watched PDF, and refreshes only those rendered pages. The browser
is never reloaded automatically: it shows ``Update ready`` so a batch of Codex
annotations can finish before the user applies the new revision once.

Examples:
    python3 scripts/serve_typst_review.py L1
    python3 scripts/serve_typst_review.py L1 L2 --port 8765
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from difflib import SequenceMatcher
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "scripts" / "build_typst_review_html.py"
PDF_DIR = REPO / "slides-pdf"


def source_files() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for source in sorted(REPO.glob("lecture*/L*.typ")):
        deck_id = source.stem.split("-", 1)[0].upper()
        sources[deck_id] = source
    return sources


def page_markers(lines: list[str]) -> list[int]:
    markers: list[int] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.fullmatch(r"#title-slide\(\s*\)", stripped):
            markers.append(index)
        elif stripped.startswith("#focus-slide["):
            markers.append(index)
        elif re.match(r"^==?\s+.+?\s*$", stripped):
            markers.append(index)
    return markers


def page_for_line(markers: list[int], line: int) -> int | None:
    page_index = bisect_right(markers, line) - 1
    return page_index + 1 if page_index >= 0 else None


def changed_pages(old_text: str, new_text: str) -> list[int] | None:
    """Return affected physical pages, or None when a full rebuild is safer."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    old_markers = page_markers(old_lines)
    new_markers = page_markers(new_lines)
    if not old_markers or len(old_markers) != len(new_markers):
        return None

    pages: set[int] = set()
    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_positions = range(old_start, max(old_end, old_start + 1))
        new_positions = range(new_start, max(new_end, new_start + 1))
        for position in old_positions:
            page = page_for_line(old_markers, min(position, len(old_lines) - 1))
            if page is None:
                return None
            pages.add(page)
        for position in new_positions:
            page = page_for_line(new_markers, min(position, len(new_lines) - 1))
            if page is None:
                return None
            pages.add(page)
    return sorted(pages)


def page_spec(pages: list[int]) -> str:
    if not pages:
        return ""
    ranges: list[str] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def wait_until_stable(path: Path, after_mtime_ns: int, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_since = 0.0
    while time.monotonic() < deadline:
        try:
            stat = path.stat()
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        if stat.st_mtime_ns <= after_mtime_ns:
            time.sleep(0.1)
            continue
        if stat.st_size == last_size:
            if not stable_since:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 0.25:
                return stat.st_mtime_ns
        else:
            last_size = stat.st_size
            stable_since = 0.0
        time.sleep(0.08)
    raise TimeoutError(f"Timed out waiting for {path.relative_to(REPO)}")


class DeckWorker:
    def __init__(self, deck_id: str, source: Path, dpi: int, debounce: float):
        self.deck_id = deck_id
        self.source = source
        self.pdf = PDF_DIR / f"{deck_id}.pdf"
        self.dpi = dpi
        self.debounce = debounce
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"review-{deck_id}", daemon=True
        )
        self.watch_process: subprocess.Popen[str] | None = None
        self.old_text = source.read_text(encoding="utf-8")
        self.source_mtime_ns = source.stat().st_mtime_ns
        self.pdf_mtime_ns = self.pdf.stat().st_mtime_ns if self.pdf.exists() else 0
        self.ignore_next_pdf_only_change = True

    def initial_build(self) -> None:
        subprocess.run(
            [sys.executable, str(BUILDER), "--dpi", str(self.dpi), self.deck_id],
            cwd=REPO,
            check=True,
        )

    def start(self) -> None:
        command = [
            "typst",
            "watch",
            "--root",
            str(REPO),
            "--input",
            "handout=true",
            str(self.source),
            str(self.pdf),
        ]
        self.watch_process = subprocess.Popen(command, cwd=REPO, text=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.watch_process and self.watch_process.poll() is None:
            self.watch_process.terminate()
            try:
                self.watch_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.watch_process.kill()
        self.thread.join(timeout=3)

    def _stable_source(self) -> tuple[str, int]:
        stable_mtime = self.source.stat().st_mtime_ns
        while not self.stop_event.wait(self.debounce):
            current_mtime = self.source.stat().st_mtime_ns
            if current_mtime == stable_mtime:
                return self.source.read_text(encoding="utf-8"), current_mtime
            stable_mtime = current_mtime
        return self.old_text, self.source_mtime_ns

    def _refresh_review(self, pages: list[int] | None) -> None:
        command = [
            sys.executable,
            str(BUILDER),
            "--dpi",
            str(self.dpi),
        ]
        if pages:
            command.extend(["--pages", page_spec(pages)])
        command.append(self.deck_id)
        description = f"slides {page_spec(pages)}" if pages else "all slides"
        started = time.monotonic()
        subprocess.run(command, cwd=REPO, check=True)
        elapsed = time.monotonic() - started
        print(
            f"{self.deck_id}: review update ready ({description}, {elapsed:.2f}s)",
            flush=True,
        )

    def _run(self) -> None:
        while not self.stop_event.wait(0.2):
            try:
                current_source_mtime = self.source.stat().st_mtime_ns
                if current_source_mtime == self.source_mtime_ns:
                    if self.pdf.exists():
                        current_pdf_mtime = self.pdf.stat().st_mtime_ns
                        if current_pdf_mtime > self.pdf_mtime_ns:
                            self.pdf_mtime_ns = wait_until_stable(
                                self.pdf, self.pdf_mtime_ns, timeout=45.0
                            )
                            if self.ignore_next_pdf_only_change:
                                self.ignore_next_pdf_only_change = False
                            else:
                                self._refresh_review(None)
                    continue
                new_text, new_source_mtime = self._stable_source()
                if new_text == self.old_text:
                    self.source_mtime_ns = new_source_mtime
                    self.ignore_next_pdf_only_change = True
                    continue
                pages = changed_pages(self.old_text, new_text)
                previous_pdf_mtime = self.pdf_mtime_ns
                self.pdf_mtime_ns = wait_until_stable(
                    self.pdf, previous_pdf_mtime, timeout=45.0
                )
                self.ignore_next_pdf_only_change = False
                self._refresh_review(pages)
                self.old_text = new_text
                self.source_mtime_ns = new_source_mtime
            except Exception as error:  # keep the review server alive and report the failure
                print(f"{self.deck_id}: automatic review update failed: {error}", flush=True)
                try:
                    self.source_mtime_ns = self.source.stat().st_mtime_ns
                    self.old_text = self.source.read_text(encoding="utf-8")
                except OSError:
                    pass


class QuietReviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        if self.path.startswith("/slides-review/") or self.path == "/":
            return
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decks", nargs="+", help="Deck IDs such as L1 L2 L3A")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Review rasterization resolution; 180 dpi is suitable for Retina displays",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.8,
        help="Seconds that the source must remain unchanged before rebuilding",
    )
    args = parser.parse_args()

    sources = source_files()
    deck_ids = [deck.upper() for deck in args.decks]
    missing = [deck for deck in deck_ids if deck not in sources]
    if missing:
        raise SystemExit("Typst source not found for: " + ", ".join(missing))

    workers = [
        DeckWorker(deck, sources[deck], dpi=args.dpi, debounce=args.debounce)
        for deck in deck_ids
    ]
    handler = partial(QuietReviewHandler, directory=str(REPO))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True

    for worker in workers:
        worker.initial_build()
    for worker in workers:
        worker.start()

    stopping = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    decks = ", ".join(deck_ids)
    print(
        f"Reviewing {decks} at http://127.0.0.1:{args.port}/slides-review/ "
        "(warm Typst watch; browser reload remains manual)",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        for worker in workers:
            worker.stop()
        server.server_close()


if __name__ == "__main__":
    main()
