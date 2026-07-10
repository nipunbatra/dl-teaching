"""Overflow scan for the L19 handout: render each page at 80 dpi, check the
bottom band y in [0.905H, 0.965H], x in [0.16W, 0.84W] for non-background ink.
bg is #FAFAFA (250,250,250) light or dark #23373B. >4% non-bg => likely overflow.
Also flags duplicate heading text = silent 2-page spillovers (handled by caller via pdftotext).
Usage: python3 lecture19/diagrams/overflow_scan.py /tmp/L19h_pages
"""
import sys, glob, os
from PIL import Image

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, '*.png')), key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or 0))
bad = []
for f in files:
    im = Image.open(f).convert('RGB'); W, H = im.size
    px = im.load()
    x0, x1 = int(0.16*W), int(0.84*W)
    y0, y1 = int(0.905*H), int(0.965*H)
    tot = 0; nonbg = 0
    for y in range(y0, y1):
        for x in range(x0, x1, 2):
            r, g, b = px[x, y]; tot += 1
            light = abs(r-250) <= 6 and abs(g-250) <= 6 and abs(b-250) <= 6
            dark  = abs(r-35) <= 10 and abs(g-55) <= 10 and abs(b-59) <= 10
            white = r >= 246 and g >= 246 and b >= 246
            if not (light or dark or white):
                nonbg += 1
    frac = nonbg / max(tot, 1)
    if frac > 0.04:
        bad.append((os.path.basename(f), round(frac*100, 1)))

if bad:
    print("POTENTIAL OVERFLOW (page, %% ink in bottom band):")
    for name, pct in bad:
        print(f"  {name}: {pct}%")
else:
    print("CLEAN — no bottom-band overflow above 4% on any page")
