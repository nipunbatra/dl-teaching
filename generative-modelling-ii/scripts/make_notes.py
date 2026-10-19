"""Write INSTRUCTOR_NOTES.md from the speaker notes in the built deck (run build_deck.py first).

    python3 scripts/make_notes.py

One entry per slide: title, then the cue for each click (<p data-say="n">) and any extra notes.
"""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(s):
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def main():
    body = (ROOT / "index.html").read_text()   # built deck: numbers already filled in
    out = ["# Instructor notes · Generative Modelling II", "",
           "Generated from the speaker notes in `deck/slides/*.html` (press **S** in the deck to see them per click).",
           "“Click n” is what to say when click n appears. Numbers come from `figures/data/numbers.json`.", ""]
    section = None
    for k, m in enumerate(re.finditer(r'<section class="slide([^"]*)" id="([^"]+)" data-section="([^"]+)">(.*?)</section>', body, flags=re.S), 1):
        cls, sid, sec, inner = m.groups()
        if sec != section:
            out += ["", f"## {sec}", ""]
            section = sec
        h2 = re.search(r"<h2>(.*?)</h2>", inner, flags=re.S)
        title = text(re.sub(r'<span class="sub">', " · ", h2.group(1))) if h2 else sid
        out.append(f"### {k}. {title}  `#{sid}`")
        notes = re.search(r'<aside class="notes">(.*?)</aside>', inner, flags=re.S)
        if notes:
            for pm in re.finditer(r'<p(?: data-say="([^"]+)")?>(.*?)</p>', notes.group(1), flags=re.S):
                say, t = pm.groups()
                out.append(f"- **click {say}:** {text(t)}" if say else f"- {text(t)}")
        q = re.search(r'<p class="q"[^>]*>(.*?)</p>', inner, flags=re.S)
        if q:
            out.append(f"- *Bottom line on the slide:* {text(q.group(1))}")
        out.append("")
    (ROOT / "INSTRUCTOR_NOTES.md").write_text("\n".join(out))
    print("INSTRUCTOR_NOTES.md:", k, "slides")


if __name__ == "__main__":
    main()
