# NLLB-200-distilled-600M (translate)

- **Repo / version:** `facebook/nllb-200-distilled-600M`
- **Parameters / size / runtime memory:** 600M params; ~2.4 GB on disk
- **Quantization available:** fp16
- **Apple Silicon (MPS):** ✅
- **CUDA:** ✅
- **Languages:** 200; Eastern Armenian as `hye_Armn`
- **License / gating:** CC-BY-NC 4.0
- **Integration surface:** `patmir/eval/candidates/translation/nllb_200_distilled_600m.py`
- **Status in pipeline:** **superseded** (was Phase 1 default; replaced by NLLB-3.3B in Phase 1.3)
- **Last evaluated:** 2026-05-24 (Phase 0)
- **Phase 0 scores** (1-5 rubric, n=40 combined EN→hy_AM + RU→hy_AM):

  | metric             | overall | cartoon | proper | narration | idiom |
  | ------------------ | ------: | ------: | -----: | --------: | ----: |
  | meaning_preserved  | 3.33    | 4.00    | 5.00   | 2.10      | 1.33  |
  | register           | 3.15    | 3.56    | 4.88   | 2.20      | 1.33  |
  | naturalness        | 3.12    | 3.38    | 4.50   | 2.60      | 1.50  |

- **Our experience:** Cleared the Phase 0 quality floor on cartoon dialogue. Idioms came out as literal calques (1.33 — unusable). RU-narration was meaningfully weaker than EN-narration. Both weak spots improved enough with NLLB-3.3B to justify the swap.
- **Notes:** Kept registered in `patmir.toml` for legacy comparison runs; not used as the active default.
