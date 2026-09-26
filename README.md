# kibitzr-maia

Code that [Kibitzr](https://kibitzr.app) uses to run the Maia-3 chess models in
the browser: the ONNX export script, and the TypeScript tokenizer and move
vocabulary that feed the exported model.

- `export/export_onnx.py` converts a Maia-3 PyTorch checkpoint to ONNX
  (fp32, fp16, int8) in a form that onnxruntime-web will run. Three constructs
  in the PyTorch model trace badly for the web runtime, so the export re-expresses
  them and gates the rewrite on numerical parity with the original forward pass.
  The script's docstrings record what was changed and why.
- `browser/tokenizer.ts` turns a FEN history (plus clocks, for the timed model)
  into the model's input tensor. It is a TypeScript port of maia3's Python
  tokenization chain.
- `browser/vocab.ts` maps between UCI moves and the 4352-entry move vocabulary
  that indexes the policy output, using closed-form index arithmetic.
  `browser/__tests__/vocab.checksum.test.ts` pins the enumeration to a frozen
  SHA-256 digest.

## Upstream

The models and the Python code they run under come from the CSSLab Maia team:

- Code: https://github.com/CSSLab/maia3 (pinned at `1e13597c`)
- Weights, 23M ponder: https://huggingface.co/UofTCSSLab/Maia3-23M-ponder
  (pinned at revision `34ec2be9`)
- Weights, 79M: https://huggingface.co/UofTCSSLab/Maia3-79M

The provenance notice Kibitzr serves alongside the model files is at
https://kibitzr.app/models/maia/NOTICE.md

Thanks to the CSSLab Maia team for the models and for publishing the code that
makes work like this possible.

## License

AGPL-3.0. This repository contains code derived from AGPL-licensed Maia-3 code,
so the same terms apply to all of it. See `LICENSE` and `NOTICE`.
