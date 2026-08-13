# Vendored Chalkdust subset

This directory contains the smallest Chalkdust source closure used by
`common/mldiag.typ`. It makes the lecture sources independent of Typst's
machine-local `@local` package registry.

## Provenance

- Upstream: <https://github.com/nipunbatra/chalkdust>
- Source checkout: `/Users/nipun/git/chalkdust` (path recorded for provenance
  only; builds do not access it)
- Upstream Git HEAD: `bb23f7c67ba13695d268caf5350b17c66b7f3d40`
- HEAD commit date: `2026-07-11T20:18:06+05:30`
- Snapshot time: `2026-08-13T10:04:50+05:30`
- Source-manifest SHA-256: `edd6aad1fab0ed9a412437ec650fd98ed474841c2ec1d6bad100653856d95d0a`
- Vendored-manifest SHA-256: `2079934fc2869f08dd3822883489a999985a3c4c24436d001fd70ac5c148ef51`

The upstream checkout had uncommitted package work at snapshot time, so the
Git commit alone does not identify these bytes. The per-file hashes below are
the authoritative snapshot. Each vendored library is byte-for-byte identical
to the source snapshot except that executable Chalkdust imports are relative
and import examples no longer point at `@local` packages.

The manifest hashes above are SHA-256 digests of a newline-delimited, sorted
list of `relative-path`, two spaces, and the corresponding file SHA-256. The
source manifest uses `convgrid/src/lib.typ`; the vendored manifest uses
`convgrid/lib.typ`.

| Package | Upstream source SHA-256 | Vendored SHA-256 |
|---|---|---|
| `autodiff` | `a3380f73f771a36c79bceda22498a8128ad6f01b401dacc533ef73555259552f` | `3fe90513e3f195279e95694112b4c3acaa04b5c88b07a87c30f2215eb0441522` |
| `bits` | `a43bb3356ab9fe7acd877743594b14aca02e386e6fe701825abe6b8ad72ca319` | `b3d5a8b6d09478ca52e7d369866130b2d2e8a971f965263f6520a6b1ac4d669a` |
| `convgrid` | `33f67d026e65dc4a98e414670a9fb47dae2d44802e079585862fba3589906ea2` | `ff06cae73202e9e763d967312bf0db657b88b1e63891231c1c9f7f8d7a8a4328` |
| `dist` | `055f29052f90b663e8b64c911d783a4234814e13f96e6dca1fae197019f3ddd1` | `47d5c9a36f093b5f0b539c67e095bfc6303c41ad2b673f1bb48b911a7d5d253d` |
| `field` | `5f7f92b0c9692894ca39961e9f1aaf930f3e510996baab8380d2896f0123b84e` | `42da0f800ef7cfbacc1466953c47d2f9d3a9aef3ccc00120b7fad0514ff5ae85` |
| `frame` | `e7d020dabcbd74340d9b027469a57c2dbc5ae376996074951956bb4e1d8054bd` | `03a10d652224db14cce2bfea87b2a48a7e34d8ab59090d9b68900995d4cffe6d` |
| `learn` | `dfe1b050700fff4774c9c90663398cea17aa2cbfd64764d94cfd0ae3af82efa0` | `5b6848b6e314760e99a34bebbb09203e1343bb5788ad24a2ae2c77e0887842db` |
| `linalg` | `d234f86b5ad83ee9307d78513fcc4a9bd18145e1766c4dc7f7109234fbc761c9` | `40243e30400018d30bd2fc1669e1b203a6eaca89a61ebf4d3dbe66352a0dfe9c` |
| `optim` | `144eab135d27b63b98783a5903641cd544a4e97bfddec50416f9c9aa7637de1f` | `ed8c1b1a25946ac945b8a52fa1003ea017a9a21589965d973eed452cbbce8eb8` |
| `plot` | `fc53865084034c5a17ac57ef862b889249123f072f5fd61a27a9a2f406996835` | `737e38e6506abc31db7baea639361141bf63841dc4cb2f370fa6b55a6affff0d` |
| `rand` | `9fadce823e594b6e5adc2bc1624b60cef1166fbc74494cb49f7887ee99b11b2d` | `8191f49e77966d76ec71f50aa19cdfffc22cf8ed90394da8f65e447531df23e8` |
| `theme` | `94d46d1287a07408b638cc0a0ddf2ba574879d7ba3d3c949241f71fd52c70288` | `512837a4e8cd1e159ff399ad69abf4ada57daa9450c3a2e42c1cc9b2c7d21619` |

The copied MIT license is `LICENSE` (SHA-256
`396e07ecd3226aa9132e0c675270d6d68d6c9642a00b42db17dcd14be3c6c0e7`).

## Dependency closure

`mldiag.typ` directly imports `theme`, `convgrid`, `plot`, `frame`, `dist`,
`field`, `optim`, `rand`, `learn`, and `autodiff`. `rand` requires `bits`, and
`learn` requires `linalg`. Chalkdust's `tensor` package is not in this closure
and is intentionally not copied.

`convgrid`, `plot`, and `field` use the published Typst package
`@preview/cetz:0.5.2`; `autodiff` uses `@preview/fletcher:0.5.8`. These are
versioned preview dependencies, not machine-local Chalkdust packages.

## Clean-local-registry check

Compile with an empty local package registry while retaining Typst's normal
preview-package cache:

```sh
empty_packages="$(mktemp -d)"
TYPST_PACKAGE_PATH="$empty_packages" typst compile --root . \
  --input handout=true lecture1/L1-probabilistic-view.typ /tmp/L1.pdf
```

With respect to Chalkdust, the dependency list must contain the files under
`vendor/chalkdust` and no `@local` Chalkdust path. The lecture theme has its
own versioned preview-package dependencies, which remain visible in that list.
