# First-10-lectures polish — joint review (me + Gemini-2.5-pro + Codex gpt-5.5/xhigh)

Per lecture: ① migrate figures to cream/anthropic palette if old · ② Gemini deep review · ③ Codex correctness review (`codex exec --sandbox read-only`, gpt-5.5 xhigh — direct CLI; the multi-llm MCP is broken) · ④ fix overflow/examples/errors · ⑤ build + PDF spot-check · ⑥ commit.

- ✅ L00 · probability/MLE — polished (multiple rounds)
- ✅ L00B · Bayes/MAP/reg — polished
- ✅ L00C · information theory — polished + deepened + gemini-reviewed
- ✅ L01 · why DL / MLP / backprop — cream figs, trimmed, reviewed
- ✅ L02 · UAT / going deep — cream figs migrated, gemini+codex reviewed & fixed
- ⬜ L03 · training in practice
- ⬜ L04 · SGD / momentum
- ⬜ L05 · Adam / schedules
- ⬜ L06 · regularization
- ⬜ L07 · CNN deep dive
- ⬜ L08 · modern CNNs / transfer
- ⬜ L09 · detection / segmentation
- (L10 · RNN/LSTM/GRU — stretch)

Tooling fixed: `codex exec --sandbox read-only "<prompt>"` from repo root uses gpt-5.5 @ xhigh (config.toml). Gemini via mcp gemini_analyze = gemini-2.5-pro (best large-context).
