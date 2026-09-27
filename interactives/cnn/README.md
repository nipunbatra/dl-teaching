# CNNs, from first principles

One unified interactive slide deck for ES 667.

[Read online](https://nipunbatra.github.io/dl-teaching/interactives/cnn/index.html) ·
[Present](https://nipunbatra.github.io/dl-teaching/interactives/cnn/index.html?present)

Open `index.html`
through a local server. No build step or external runtime/CDN is
needed; all browser assets are in this folder. Internet is needed only for
external reference links.

From the DL teaching repository root:

```sh
python3 -m http.server 8792 --bind 127.0.0.1
# http://127.0.0.1:8792/interactives/cnn/index.html
# http://127.0.0.1:8792/interactives/cnn/index.html?present
```

The photo calculator uses canvas and needs HTTP (a local server is enough).
Other worksheets also work when opening the HTML files directly.

## Content

- Inductive bias, real image responses, editable convolution, padding,
  stride, dilation, the output-size derivation, structured linear maps.
- Channel reduction, 1×1 mixing, ReLU/pooling, shift tests, receptive
  fields and support holes, parameter/MAC/activation accounting.
- The ML LeNet ledger, DL L8's exact shared gradients, a live trainable
  26-parameter CNN, the same model's
  forward trace, PyTorch, network ledger, depthwise/residual/transfer bridges.

There are 31 classroom frames in one sequence, with one title, one overview,
and one shared interactive state. Allow approximately 100–130 minutes with
interaction; architecture bridges can be assigned as reading. Source selection,
attribution, and evidence limits are in `sources.html`.

## Classroom controls

P / Present enters a 16:9 stage; append `?present#convolution` for a deep link.
Left/right or Page Up/Down change frames; N advances even from a focused
slider. O opens an overview. S opens the current teaching notes. Escape closes
dialogs first, then exits presentation. Sliders retain native arrow keys;
Escape releases control focus. Reading mode is responsive, and the numerical
widgets are the same DOM objects in both modes. Closing a presentation does
not reset the model. Training resets on a page reload and is not persisted.

## Files

- `index.html`: the single directly editable lecture source.
- `part1.html`, `part2.html`, `part3.html`: tiny compatibility redirects for
  previous preview links, with no duplicate lecture content.
- `style.css`: shared layout and fixed 1280×720 stage.
- `model.js`: pure functions, browser and Node; numerical source of truth.
- `app.js`: controls, widgets, and navigation.
- `initial-model.json`: exact seed-7 initial weights and six training images.
- `tiny_cnn.py`: float64 PyTorch forward/gradient parity and training companion.
- `figures/`: two existing Oxford-IIIT Pet derivatives, CC BY-SA 4.0.

No training dataset, pretrained model, UI framework, or new npm dependency is
downloaded. The existing repo Puppeteer dependency is used only for checks.

## Verification

Run from the repository root:

```sh
node scripts/check_cnn_math.cjs
node scripts/check_cnn_browser.cjs
python3 interactives/cnn/tiny_cnn.py --check
```

The browser check starts its own loopback server, exercises widgets and
keyboard behavior, checks all frames and opened answers for clipping, checks
mobile reading layout, and saves screenshots in `tmp/cnn-review/`.
PyTorch is needed only for the optional Python companion.
This machine's existing cached environment can run it with
`uv run --offline --with torch python interactives/cnn/tiny_cnn.py --check`.

Numerical checks include known convolution answers, stride/dilation geometry,
independent scalar references, shared/input gradients, finite differences for
all 26 trainable parameters, shift counterexamples, receptive fields, costs,
training improvement, and the published initialization. No test-set accuracy
claim is made.

## Provenance

Local source decks: `lecture8/L8-cnns.typ`, `lecture8b/L8b-modern-cnns.typ`, and
`ml-teaching/neural-networks/assets/cnn-notes.pdf` plus its convolution/stride/
edge notebooks. The original decks remain intact. Attention parts I–III
informed pacing, semantic color, prediction/reveal, and article/classroom modes.

Images copied without modification from
`shared/vision-evidence/oxford-iiit-pet/derived/`. See that directory's
`evidence.json` and parent README for hashes, crop coordinates, and provenance.
The browser's resize/filter transform is documented in `sources.html`.
