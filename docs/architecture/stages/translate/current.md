# Active translate config

```toml
[pipeline]
translate = "nllb_200_3.3b"

[[eval.translation.candidates]]
id        = "nllb_200_3.3b"
enabled   = true
module    = "patmir.eval.candidates.translation.nllb_200_distilled_600m"  # same loader
model_id  = "facebook/nllb-200-3.3B"
device    = "mps"
dtype     = "float32"
max_new_tokens = 512
num_beams = 4
no_repeat_ngram_size = 3
```

- Same loader as the 600M distilled candidate (`nllb_200_distilled_600m.py`); the only difference is `model_id` and dtype.
- Routes by `Transcript.language`: `en` → `eng_Latn` source; `ru` → `rus_Cyrl` source. Target is always `hye_Armn`.
- ~6.6 GB resident on MPS at fp32. Fits the 32 GB Mac comfortably.

## Phase 2.x

The emotion-transfer spec adds an `emotion` field to each `Sentence` and `TranslatedSentence`. NLLB has no prompt surface — it ignores the field, no behavior change. The emotion gets used only when a prompt-driven translator (Qwen3-Instruct-MLX) is the active candidate.
