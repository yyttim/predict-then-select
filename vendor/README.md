# Reference model

`reference_model.py` is copied without modification from
[Single-Cell-Phenotyping](https://github.com/saqibnaziir/Single-Cell-Phenotyping),
commit `56a42e77bf6a7c5fcbbf8967dc209e2ee8f3d5df`. Its MIT license is retained
in [LICENSE](LICENSE).

The associated paper is Saqib Nazir and Ardhendu Behera,
*Towards Label-Free Single-Cell Phenotyping Using Multi-task Learning*,
[DOI: 10.1007/978-3-032-31666-0_41](https://doi.org/10.1007/978-3-032-31666-0_41).

## Adaptation

The encoder in `src/cellselect/model.py` preserves the released ECA attention,
CNN, ViT, and weighted-addition fusion. The CNN has two residual Inception
modules; the ViT uses 4-by-4 patches, width 128, two blocks, and four heads.
The shared representation has width 256.

Task refinements and the 128/64-unit prediction heads are retained. Cross-task
gating is replaced by independent heads that interact through the shared encoder.
The regression head predicts one marker. Classification returns logits, and
marker normalization is handled by the data pipeline.

The adapted model has 1,289,713 parameters: 941,165 in the encoder, 174,339 in
the classification head, and 174,209 in the regression head. Dropout rates match
the reference. Attention weights are not materialized during the forward pass.

Tests load identical weights into the adaptation and the ungated reference,
compare outputs and features, and check gradient flow through both encoder
branches. The vendor file is used by these tests, not imported at runtime.

| Preserved file | SHA-256 |
| --- | --- |
| `reference_model.py` | `d3a7fc2e363d4715a36388f051333812bd284a69e530d2368f2f5eba5fc0a7a9` |
| `LICENSE` | `2d078ab42b19028e4f57fdf8f78740ca6b8a5b5ef009f61c160f0469db1e7781` |
