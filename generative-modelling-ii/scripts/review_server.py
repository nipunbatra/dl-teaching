"""Serve the deck locally and collect right-click review notes.

    python3 scripts/review_server.py            # http://127.0.0.1:8772/

- Static files from the repository root, never cached, so edits show up immediately.
- GET/POST /api/notes?deck=part1.html  : the deck's notes, stored in review/notes.json
- Every new or edited note is also appended as one JSON line to review/inbox.jsonl,
  which Claude watches, so a right-click note reaches the session as soon as you save it.
- GET /api/version : changes whenever a built deck file changes; open pages reload themselves.
Only listens on 127.0.0.1.
"""
import argparse
import json
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review"
NOTES = REVIEW / "notes.json"
INBOX = REVIEW / "inbox.jsonl"
WATCH = ["index.html", "deck/deck.css", "deck/deck.js", "deck/widgets.js", "deck/annotate.js"]
LOCK = threading.Lock()
CLAUDE_FIELDS = ("status", "reply", "done_at")  # written by Claude in notes.json, never overwritten by the page


def load():
    try:
        return json.loads(NOTES.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(all_notes):
    REVIEW.mkdir(exist_ok=True)
    tmp = NOTES.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_notes, indent=1, ensure_ascii=False))
    tmp.replace(NOTES)


def version():
    return str(max((ROOT / f).stat().st_mtime_ns for f in WATCH if (ROOT / f).exists()))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/version":
            return self._json({"version": version()})
        if url.path == "/api/notes":
            deck = parse_qs(url.query).get("deck", ["index.html"])[0]
            with LOCK:
                return self._json(load().get(deck, []))
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/notes":
            return self._json({"error": "not found"}, 404)
        deck = parse_qs(url.query).get("deck", ["index.html"])[0]
        incoming = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "[]")
        with LOCK:
            all_notes = load()
            old = {n["id"]: n for n in all_notes.get(deck, [])}
            merged, events = [], []
            for n in incoming:
                prev = old.get(n["id"])
                for f in CLAUDE_FIELDS:  # Claude's status and reply win over the page's copy
                    if prev and prev.get(f) is not None:
                        n[f] = prev[f]
                if prev is None:
                    events.append(("new", n))
                elif prev.get("note") != n.get("note"):
                    n["status"] = "open"  # edited after Claude answered: needs another look
                    events.append(("edited", n))
                merged.append(n)
            all_notes[deck] = merged
            save(all_notes)
            if events:
                with INBOX.open("a") as fh:
                    for kind, n in events:
                        fh.write(json.dumps({
                            "event": kind, "deck": deck, "id": n["id"], "slide": n.get("slide"),
                            "slideNo": n.get("slideNo"), "build": n.get("build"), "title": n.get("title"),
                            "clicked": n.get("target"), "note": n.get("note"),
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        }, ensure_ascii=False) + "\n")
        return self._json(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8772)
    a = ap.parse_args()
    REVIEW.mkdir(exist_ok=True)
    INBOX.touch()
    print(f"Review server: http://127.0.0.1:{a.port}/  (notes → {NOTES.relative_to(ROOT)})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
