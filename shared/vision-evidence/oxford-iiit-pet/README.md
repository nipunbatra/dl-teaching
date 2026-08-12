# Oxford-IIIT Pet teaching evidence

This directory contains one real, ground-truth-annotated sample selected as a
shared evidence case for public Lectures 8–11.

## Sample

- Dataset: Oxford-IIIT Pet
- Image id: `Abyssinian_1`
- Original image: `Abyssinian_1.jpg` (`600 × 400`)
- Ground-truth trimap: `Abyssinian_1-trimap.png` (`600 × 400`)
- Ground-truth head ROI: `Abyssinian_1.xml`
  - `(xmin, ymin, xmax, ymax) = (333, 72, 425, 158)`
- Dataset page: <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- Dataset paper: Parkhi et al., *Cats and Dogs*, CVPR 2012
- License: CC BY-SA 4.0; copyright remains with the original image owner.

The official dataset page states that every image has a breed/species label, a
tight head ROI, and a pixel-level foreground/background trimap. The image was
retrieved from a CC BY-SA mirror of the official dataset, and its XML/trimap
were extracted from the official annotation archive.

## File hashes

- `Abyssinian_1.jpg`:
  `2533197401eebe9410ea4d063f86c43fbd2666f3e8165a38aca155c0d09c21be`
- `Abyssinian_1-trimap.png`:
  `a39ce8ec0363178918cb257844125e3aa7773a3acd4ca8b84780d62c1fa6f220`
- `Abyssinian_1.xml`:
  `bea13ddcf1dc171e39da5ce2c134b1dc8c247a9f1bb21cc2056f37f7a423a51f`

## Teaching contract

The photograph and annotations are evidence, not decoration:

- Lecture 8 may compute real grayscale/RGB patches and convolution responses.
- Lecture 9 may show preprocessing/crop/feature behavior or a reproducible
  pretrained-model probe, clearly distinguishing observation from schematic.
- Lecture 10 may use the actual XML head box and then contrast it with a
  detector's scored predictions/NMS.
- Lecture 11 may use the actual trimap and derived foreground mask for semantic
  metrics, then explain why a one-pet image does not demonstrate instance
  separation by itself.

Any derived crop, overlay, activation, box, or mask must record the generating
script, exact model/checkpoint when applicable, and whether it is ground truth,
model output, or a pedagogical construction.
