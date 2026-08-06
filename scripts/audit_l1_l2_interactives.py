#!/usr/bin/env python3
"""Audit interactive links embedded in the Lecture 1 and Lecture 2 Typst decks.

The default (offline) audit extracts every ``#interbox(link-to: ...)`` target,
resolves simple string constants such as ``IA + "slug"``, and checks any
target backed by this repository.  ``--live`` additionally requests every
HTTP(S) target and requires a successful HTML response.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECKS = (
    Path("lecture1/L1-probabilistic-view.typ"),
    Path("lecture2/L2-linear-to-mlp.typ"),
)

LET_STRING_RE = re.compile(
    r'^\s*#let\s+([A-Za-z_][\w-]*)\s*=\s*"([^"\n]*)"', re.MULTILINE
)
INTERBOX_RE = re.compile(
    r"#interbox\s*\(\s*link-to\s*:\s*([^\n\)]*)\)", re.MULTILINE
)
STRING_TOKEN_RE = re.compile(r'^"([^"\n]*)"$')
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][\w-]*$")


@dataclass(frozen=True)
class LinkRef:
    deck: Path
    line: int
    expression: str
    url: str


@dataclass(frozen=True)
class CheckResult:
    ref: LinkRef
    status: str
    detail: str
    failed: bool = False


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all #interbox links in the Lecture 1 and Lecture 2 Typst "
            "decks. Local validation is always performed; --live also checks "
            "published URLs."
        )
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="request every HTTP(S) link and require a 2xx HTML response",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="per-request timeout in live mode (default: 10)",
    )
    return parser.parse_args(argv)


def resolve_expression(expression: str, constants: dict[str, str]) -> str:
    """Resolve the string-concatenation subset used by link-to expressions."""

    pieces: list[str] = []
    for raw_token in expression.split("+"):
        token = raw_token.strip()
        string_match = STRING_TOKEN_RE.fullmatch(token)
        if string_match:
            pieces.append(string_match.group(1))
        elif IDENTIFIER_RE.fullmatch(token) and token in constants:
            pieces.append(constants[token])
        else:
            raise ValueError(
                f"unsupported link expression token {token!r}; "
                "use a quoted string or a #let string constant"
            )
    return "".join(pieces)


def extract_links(deck: Path) -> list[LinkRef]:
    absolute_deck = REPO_ROOT / deck
    try:
        source = absolute_deck.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read {deck}: {exc}") from exc

    constants = {match.group(1): match.group(2) for match in LET_STRING_RE.finditer(source)}
    refs: list[LinkRef] = []
    for match in INTERBOX_RE.finditer(source):
        expression = match.group(1).strip()
        line = source.count("\n", 0, match.start()) + 1
        try:
            url = resolve_expression(expression, constants)
        except ValueError as exc:
            raise RuntimeError(f"{deck}:{line}: {exc}") from exc
        refs.append(LinkRef(deck=deck, line=line, expression=expression, url=url))
    return refs


def repository_path_for(url: str) -> Path | None:
    """Map repository-owned public URLs to their source artifact, if any."""

    parsed = urlparse(url)
    path = unquote(parsed.path)

    if not parsed.scheme:
        candidate = REPO_ROOT / path.lstrip("/")
    elif parsed.netloc == "github.com":
        match = re.fullmatch(
            r"/nipunbatra/dl-teaching/blob/[^/]+/(.+)", path, re.IGNORECASE
        )
        if not match:
            return None
        candidate = REPO_ROOT / match.group(1)
    elif parsed.netloc == "colab.research.google.com":
        match = re.fullmatch(
            r"/github/nipunbatra/dl-teaching/blob/[^/]+/(.+)", path, re.IGNORECASE
        )
        if not match:
            return None
        candidate = REPO_ROOT / match.group(1)
    elif parsed.netloc == "nipunbatra.github.io" and path.startswith("/dl-teaching/"):
        candidate = REPO_ROOT / path.removeprefix("/dl-teaching/")
    else:
        return None

    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"mapped path escapes repository: {url}") from exc
    return resolved


def validate_local(ref: LinkRef) -> CheckResult:
    parsed = urlparse(ref.url)
    if parsed.scheme not in ("", "http", "https"):
        return CheckResult(ref, "FAIL", f"unsupported URL scheme {parsed.scheme!r}", True)
    if parsed.scheme in ("http", "https") and not parsed.netloc:
        return CheckResult(ref, "FAIL", "HTTP(S) URL has no host", True)

    try:
        local_path = repository_path_for(ref.url)
    except RuntimeError as exc:
        return CheckResult(ref, "FAIL", str(exc), True)

    if local_path is None:
        return CheckResult(ref, "SKIP", "external target; use --live to verify")
    if not local_path.is_file():
        return CheckResult(
            ref,
            "FAIL",
            f"repository artifact is missing: {local_path.relative_to(REPO_ROOT)}",
            True,
        )

    suffix = local_path.suffix.lower()
    try:
        if suffix == ".ipynb":
            notebook = json.loads(local_path.read_text(encoding="utf-8"))
            if not isinstance(notebook.get("cells"), list):
                raise ValueError("top-level 'cells' is not a list")
            detail = f"valid notebook: {local_path.relative_to(REPO_ROOT)}"
        elif suffix in (".html", ".htm"):
            prefix = local_path.read_text(encoding="utf-8", errors="replace")[:4096].lower()
            if "<html" not in prefix and "<!doctype html" not in prefix:
                raise ValueError("file does not contain an HTML document header")
            detail = f"valid HTML: {local_path.relative_to(REPO_ROOT)}"
        else:
            detail = f"artifact exists: {local_path.relative_to(REPO_ROOT)}"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CheckResult(
            ref,
            "FAIL",
            f"invalid {local_path.relative_to(REPO_ROOT)}: {exc}",
            True,
        )
    return CheckResult(ref, "PASS", detail)


def validate_live(ref: LinkRef, timeout: float) -> CheckResult:
    parsed = urlparse(ref.url)
    if parsed.scheme not in ("http", "https"):
        return CheckResult(ref, "SKIP", "not an HTTP(S) target")

    request = Request(
        ref.url,
        headers={
            "User-Agent": "dl-teaching-interactive-audit/1.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            content_type = response.headers.get_content_type().lower()
            final_url = response.geturl()
            response.read(512)
    except HTTPError as exc:
        return CheckResult(ref, "FAIL", f"HTTP {exc.code}: {exc.reason}", True)
    except URLError as exc:
        return CheckResult(ref, "FAIL", f"network error: {exc.reason}", True)
    except TimeoutError:
        return CheckResult(ref, "FAIL", f"timed out after {timeout:g}s", True)

    if not 200 <= status < 300:
        return CheckResult(ref, "FAIL", f"HTTP {status} from {final_url}", True)
    if content_type not in ("text/html", "application/xhtml+xml"):
        return CheckResult(
            ref,
            "FAIL",
            f"HTTP {status}, but content type is {content_type!r} (expected HTML)",
            True,
        )
    redirect = f" -> {final_url}" if final_url != ref.url else ""
    return CheckResult(ref, "PASS", f"HTTP {status}, {content_type}{redirect}")


def print_result(prefix: str, result: CheckResult) -> None:
    location = f"{result.ref.deck}:{result.ref.line}"
    print(f"[{result.status}] {prefix:<5} {location}")
    print(f"       {result.ref.url}")
    print(f"       {result.detail}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2

    refs: list[LinkRef] = []
    try:
        for deck in DEFAULT_DECKS:
            refs.extend(extract_links(deck))
    except RuntimeError as exc:
        print(f"[FAIL] extract {exc}", file=sys.stderr)
        return 1

    if not refs:
        print("[FAIL] extract no #interbox(link-to: ...) targets found", file=sys.stderr)
        return 1

    failures = 0
    print(f"Auditing {len(refs)} L1/L2 interactive links from {len(DEFAULT_DECKS)} decks")
    for ref in refs:
        local_result = validate_local(ref)
        print_result("local", local_result)
        failures += int(local_result.failed)
        if args.live:
            live_result = validate_live(ref, args.timeout)
            print_result("live", live_result)
            failures += int(live_result.failed)

    mode = "local + live" if args.live else "local (external URLs skipped)"
    if failures:
        print(f"\nFAIL: {failures} check(s) failed in {mode} mode.")
        return 1
    print(f"\nPASS: all checks passed in {mode} mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
