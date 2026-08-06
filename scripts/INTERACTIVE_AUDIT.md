# L1/L2 interactive-link audit

The audit reads the Typst sources for Lectures 1 and 2 and checks every
`#interbox(link-to: ...)` target. It has no third-party Python dependencies.

Run the fast local audit before rebuilding the decks:

```sh
python3 scripts/audit_l1_l2_interactives.py
```

This checks repository-backed HTML and notebook files. External sites are
listed as `SKIP`, because the default mode does not use the network.

Before publishing, check every live URL with a 10-second per-request timeout:

```sh
python3 scripts/audit_l1_l2_interactives.py --live --timeout 10
```

The same checks are available through npm:

```sh
npm run audit:interactives
npm run audit:interactives:live
```

The command exits with status 1 for unresolved Typst link expressions, missing
or malformed local artifacts, HTTP errors such as 404, timeouts, or a live
response that is not HTML. This makes the live command suitable for CI or a
pre-publish checklist.
