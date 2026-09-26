# browser/

TypeScript that prepares input for the Maia-3 ONNX model and reads its policy
output. These files ship, compiled, in the Kibitzr frontend bundle; this is
their source form.

- `tokenizer.ts` is a port of maia3's Python tokenization chain
  (`maia3.dataset.tokenize_board` plus `maia3.dataset.get_historical_tokens`),
  so the browser builds the same input tensor the Python model would. It
  supports the timed 23M ponder layout (100 columns) and the untimed 79M
  layout (97 columns).
- `vocab.ts` reproduces the 4352-entry move vocabulary in closed form and
  handles the board mirroring used when Black is to move.
- `__tests__/vocab.checksum.test.ts` (vitest) enumerates every vocabulary
  index and checks the digest against the frozen `VOCAB_SHA256`.

Neither file has runtime dependencies.
