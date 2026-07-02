# data/lora_training/ — STALE (Exp 4 era, pre-cleaning)

**Do NOT train from the `*.jsonl` files in this directory.**

These are the **old, pre-cleaning LoRA training sets** used for the contaminated-data
adapters in **Experiment 4** (superseded). They predate the corpus-cleaning pipeline
introduced in Experiment 9 (front-/back-matter slicing, non-philosophical chunk
filtering, footnote stripping) and carry the same contamination that made the Exp 8
CAA vectors null.

**Clean LoRA trains on `data/chunked/`** instead — per-author chunked JSON
(`data/chunked/{author}/*.json`): Marcus 437 chunks (*Meditations*), Seneca 540
(*Moral Letters*), Epictetus 123 (*Enchiridion* only). Those are the sources for the
canonical `lora_{author}_clean` adapters (Exp 11).

Retained here only for **provenance / reproducibility of the superseded Exp 4 result**.
