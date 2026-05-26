# OPUS-MT (translate)

- **Repo / version:** `Helsinki-NLP/opus-mt-en-hy`, `Helsinki-NLP/opus-mt-ru-hy` (2020 OPUS-MT)
- **Parameters / size / runtime memory:** ~80 MB each
- **Apple Silicon (MPS):** ✅
- **Languages:** Eastern Armenian explicitly tagged (`hye`)
- **License / gating:** CC-BY 4.0
- **Integration surface:** `patmir/eval/candidates/translation/opus_mt.py`
- **Status in pipeline:** **registered, not active**
- **Last evaluated:** 2026-05-24 (Phase 1 registration, not yet scored against the masha clip)
- **Notes:**
  - BLEU ~16.6 on Tatoeba en→hy. Modest quality but tiny + fast — a useful A/B candidate without needing GPU.
  - One model per language pair (EN→hy and RU→hy are separate candidates).
