# SOTA review brief

**Purpose:** a single self-contained input for a research pass asking *"is this pipeline
still state of the art?"* Written 2026-08-24.

This file is **inventory only**. It states what Patmir is, the constraints any
replacement model must satisfy, exactly what runs today, and — the bulk of it — every
model and approach already tried and rejected, with the reason. It deliberately contains
no recommendations and no candidate shortlist: those are the research pass's output, not
its input.

**How to use it:** read §2 first. Most "better" models die on those constraints before
quality is even discussed. Then check §6 and §7 before proposing anything — a model
listed there has already been evaluated, and re-opening it needs new evidence, not a new
opinion.

`patmir.toml` is the single source of truth for what runs. Where any doc disagrees with
it, `patmir.toml` wins; known disagreements are catalogued separately in
[doc-drift.md](doc-drift.md).

---

## §1 What Patmir is

A **local-first video dubbing pipeline**: foreign-language video in (Russian primary,
English secondary) → Eastern Armenian (`hy_AM`) dub out, delivered as an MKV with
Armenian default audio, the original track retained, and an Armenian SRT.

Built by the author for his hearing-aid-wearing son. Explicitly a **workstation tool,
not a SaaS** — a polished UI is not a goal.

**Nine swappable stages:**

```
ingest → separate → diarize → emotion → asr → translate → tts → mux → output
```

Each stage is a `typing.Protocol` in `patmir/pipeline/stage.py`; exactly one concrete
candidate runs per stage, selected **by id in `patmir.toml [pipeline]`**. Model swaps are
config edits, not code changes — that is the central architectural bet, and it holds:
**74 registered candidates across the 9 stages**.

**Two entry points**, both feeding the same pipeline:

- CLI on a local file — `patmir dub <video>`
- Chrome MV3 extension → local FastAPI server → yt-dlp pre-download → same pipeline

**Two execution models**, selected per submission by a `mode` field that folds into the
cache key:

- **Batch / quality mode (D1, default)** — all stages strictly in order over the whole
  video, one cached `dub.wav`. Best quality, one wait, instant rewatch.
- **Watch-along mode (D3, additive)** — the dub is produced in VAD-snapped sections and
  streamed behind a short buffer while the viewer watches. Cross-section speaker identity
  is ECAPA-TDNN cosine matching so a character keeps one voice across sections and seeks.

**Size:** ~20,600 LOC Python across 121 files, 317 tests in 25 files, 25 ADRs.

---

## §2 Constraints any replacement must satisfy

These are the filter. Most alternatives in §6 died here, not on quality.

1. **Eastern Armenian only.** Target dialect is `hy_AM` (ISO 639-3 `hye`). Western
   Armenian (`hyw`) output is automatically rejected — it is a product-scope decision
   tied to the target user, and it does not revisit. Most multilingual engines do **not**
   distinguish the two dialects in their language tags, so any model that merely
   "supports Armenian" must be **ear-verified by the author** (a fluent EA speaker)
   before it counts. If it sounds Western, it is rejected outright.

2. **Local-first strict.** Cloud APIs are never enabled by default. Cloud is a per-stage
   opt-in, permitted only after the local set for that stage has been judged inadequate,
   and it requires an ADR. Today **`translate` is the only cloud stage**.

3. **Two machines, both memory-constrained.**
   - M1 Pro, 32 GB unified memory, MLX/MPS. All model stages MLX-native by default.
   - RTX 3090, 24 GB, on WSL2 — where the **Windows commit limit, not the card**, sets
     the ceiling. CUDA under WSL2 charges host commit ~1.06 GB per GB of VRAM, so
     `commit limit = RAM + pagefile` caps GPU allocation regardless of VRAM. On a 4 GB
     pagefile the usable budget was ~16 GB on a 24 GB card.

4. **TTS must do three things simultaneously**: production-quality Eastern Armenian,
   **zero-shot per-character voice cloning** from a reference clip, and **controllable
   emotion**. This triple requirement is what eliminated nearly every TTS alternative —
   several models are strong on two of the three.

5. **ASR must emit word-level timestamps**, or a forced aligner must cover the source
   language. Timings drive ref-clip slicing, speaker tagging, and mux placement.

6. **Non-commercial licences are acceptable** (personal tool). CC-BY-NC and
   research-only checkpoints are in scope.

7. **Diarization caps at 4 speakers.** Sortformer's architecture, on both machines.
   Content with more than 4 distinct speakers under-clusters.

8. **Source languages are Russian and English.** Russian is the validated path and the
   only language in the current eval clip.

---

## §3 Code stack

| Layer | Choice |
| --- | --- |
| Language | Python `>=3.11` (`.python-version` pins 3.11) |
| Build / lock | `hatchling` backend; **`uv`** lock, 260 packages |
| CLI | stdlib **argparse** — `patmir dub / runs / ui / bench`, plus `--profile`, `--resume-from`, `--resume-job`, `--source-lang`. No click/typer. |
| Server | **FastAPI + uvicorn**, Jinja2 templates, SSE progress via `StreamingResponse`, CORS middleware, bound `127.0.0.1:8765` |
| Config | stdlib **tomllib** over `patmir.toml`; dataclasses for resolved config and every inter-stage contract; `typing.Protocol` + `@runtime_checkable` per stage. No pydantic. |
| Lint / types | **none configured** — `[tool.hatch.build.targets.wheel]` is the only `[tool.*]` table in `pyproject.toml` |
| Tests | **pytest** `>=8` — 25 files, 317 `def test_` functions, `shutil.which("ffmpeg")` skip guards |
| Extension | Chrome **MV3, vanilla JS**, no bundler and no `package.json` (~1,300 LOC across 7 files) |
| System binaries | `ffmpeg`, `ffprobe`, `rubberband` |

### Dependency extras

Root `dependencies` is **deliberately empty** — no heavy ML deps at the root; stages pull
theirs via extras.

- **`[local]`** — `torch>=2.2`, **`transformers>=5.13`**, `sentencepiece`, `numpy>=1.26`,
  `piper-tts>=1.4`, `huggingface-hub>=0.20`, `faster-whisper>=1.0`,
  `llama-cpp-python>=0.3`, `yt-dlp`, `pyrubberband>=0.3`, `soundfile>=0.12`,
  `torchaudio>=2.2`, `demucs>=4.0`, `silero-vad>=5.0`, `pyannote.audio>=3.1`,
  `funasr>=1.1`, `modelscope>=1.20`, `speechbrain>=1.1`, plus five **Darwin-gated**
  (`; sys_platform == 'darwin'`) MLX packages: `mlx-whisper>=0.4`, `mlx-audio>=0.4`,
  `demucs-mlx>=1.4`, `mlx-qwen3-asr>=0.3`, `mlx-lm>=0.31`.
- **`[cloud]`** — `httpx>=0.27`, `anthropic>=0.30`, `openai>=1.30`,
  `google-cloud-texttospeech>=2.16`.
- **`[cuda]`** — `transformers>=5.13`, `nemo_toolkit[asr]>=2.0`. Deliberately **not** in
  `[all]`: NeMo is a heavy Linux-first build with no business on the Mac, where the
  candidates it serves are unselectable anyway.
- **`[ui]`** — `fastapi>=0.110`, `uvicorn>=0.27`, `jinja2>=3.1`.
- **`[dev]`** — `pytest>=8`.
- **`[all]`** = `patmir[local,cloud,ui]` — excludes `cuda`.

Install: `uv pip install -e ".[local,ui,dev]"` (Mac) / `".[local,ui,dev,cuda]"` (CUDA).

### The `transformers>=5.13` floor

Two independent floors stack:

- **≥5.5** — the Higgs TTS checkpoint declares `transformers_version: 5.5.0` and relies
  on 5.x contracts: its `tokenizer_config` gives `extra_special_tokens` as a **list** (62
  entries) where 4.x expects a dict, and its remote code passes `key_mapping` to
  `from_pretrained`.
- **≥5.13** — native `qwen3_asr` (`Qwen3ASRForConditionalGeneration` +
  `Qwen3ASRForTokenClassification` for the forced aligner). Below it the ASR stage dies
  with `KeyError: 'qwen3_asr'`.

**The official `qwen-asr` package must never be installed.** It pins
`transformers==4.57.6` *exactly* — mutually exclusive with the Higgs checkpoint — and
installing it silently resolves transformers down and breaks a working TTS stage. It is
also unnecessary since 5.13 shipped the native implementation.

Two undeclared-but-imported deps exist, both belonging to disabled candidates: Coqui
`TTS` (xtts_v2) and `omnivoice` (torch OmniVoice).

### Registry mechanics

`patmir/pipeline/config.py:_STAGE_NAMESPACE` maps stages to TOML namespaces:
`translate` → `[[eval.translation.candidates]]`, `tts` → `[[eval.tts.candidates]]` (both
shared with the eval harness and wrapped by `SentenceTranslator` / `SentenceTTS`
adapters), everything else → `[[stages.<stage>.candidates]]`. Each entry names a
`module`, whose `build(config)` factory is imported. Stage package `__init__.py` files are
empty — there is no decorator/registry-dict pattern.

**One trap worth knowing:** `_find_candidate` matches on `id` alone and **never reads
`enabled`**. `enabled = false` gates only the eval harness
(`patmir/eval/_loader.py`), not the pipeline — a pipeline run can select a disabled
candidate. So for the pipeline, "registered" simply means "present in the TOML".

### Machine profiles

`[pipeline]` is the Apple-Silicon set. `[pipeline.profiles.cuda]` overlays four stages;
`[pipeline.profiles.cuda.streaming]` is a further layer that displaces
`streaming/runner.py:streaming_config()`'s hardcoded MLX overrides. Selected by
environment, **never autodetected**:

```bash
export PATMIR_PROFILE=cuda          # or: patmir bench <clip> --profile cuda
```

Resolution order, last wins:

```
[pipeline]  <  [pipeline.profiles.<p>]  <  code overrides  <  [pipeline.profiles.<p>.<layer>]
```

An unknown profile name is a **hard error**, not a fall back. Profiles change `entries`,
so each profile has its own cache key and the two machines cannot serve each other a dub
built from a different model set.

---

## §4 Active model set

Two machine profiles. `patmir.toml [pipeline]` is the Mac; `[pipeline.profiles.cuda]`
overlays `separate`, `diarize`, `asr`, `tts`.

| Stage | Mac (MLX, default) | RTX 3090 (`cuda` profile) |
| --- | --- | --- |
| **ingest** | `ffmpeg` — ffmpeg + ffprobe, 16 kHz mono PCM, cpu | same |
| **separate** | `htdemucs_mlx_vocals` — `htdemucs_ft`'s **vocals-specialist sub-model only** (sig `04573f0d`) via `demucs-mlx`; `accompaniment = mix − vocals`; then Silero VAD split into `speech_vocals` / `non_speech_vocals` / `vad_intervals` | `htdemucs_vocals` — torch twin, same sub-model sig, resolved through `demucs.pretrained.get_model` |
| **diarize** | `sortformer_mlx` — `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32` via `mlx_audio.vad`. End-to-end neural, no clustering knobs; emits concurrent speakers so the candidate re-derives overlap intervals | `sortformer_nemo` — `nvidia/diar_streaming_sortformer_4spk-v2.1` on NVIDIA's own NeMo (Sortformer's reference implementation; mlx-audio is the third-party Apple port). Post-processing shared with the MLX candidate in `_postprocess.py` so the runtimes cannot drift |
| **emotion** | `emotion2vec_plus_base` — `iic/emotion2vec_plus_base` via funasr, `hub="hf"`, 9 classes → 5 bins, confidence floor 0.5 → neutral | same (device `auto` resolves to cuda) |
| **asr** | `qwen3_asr_1_7b_mlx_8bit` — `mlx-community/Qwen3-ASR-1.7B-8bit` + aligner `Qwen/Qwen3-ForcedAligner-0.6B`, via `mlx-qwen3-asr` | `qwen3_asr_1_7b` — `Qwen/Qwen3-ASR-1.7B-hf` + `Qwen/Qwen3-ForcedAligner-0.6B-hf` through transformers' **native** `qwen3_asr`. The `-hf` suffix is load-bearing. Long-form segmentation lives in the candidate: chunks cut at Silero VAD silences, capped `chunk_seconds=28`, run batched (`batch_size=8`) through ASR and aligner alike |
| **translate** | `openrouter_gemini_flash` — `google/gemini-3.5-flash` via OpenRouter, casual-register system prompt + per-sentence emotion hint, `reasoning.effort=minimal`, temp 0.2 / top_p 0.9. **The only cloud stage.** Requires `OPENROUTER_API_KEY` | same |
| **tts** | `higgs_v3_mlx_q8` — Higgs Audio v3 TTS 4B (`multimodalart/higgs-audio-v3-tts-4b-transformers` + codec `bosonai/higgs-audio-v2-tokenizer`). A **hand-written hybrid port**: MLX `mlx_lm` Qwen3-4B backbone at q8 for the AR decode + Higgs' fused multi-codebook embed/head + delay sampler; PyTorch codec on MPS for ref-encode and waveform-decode. temp 0.8 / top_p 0.95 / `max_retries=2` | `higgs_v3_torch_bf16` — same model, bf16 native, `attn_implementation="sdpa"`. Reuses the checkpoint's own delay-pattern helpers and `_sampler_step` but owns the decode loop so it runs **batched across sentences** (`max_batch=8`), chunks sorted by expected length. Batch-vs-single equivalence verified bitwise on a tiny CPU model |
| **mux** | `ffmpeg_pad_stretch` — ffmpeg + pyrubberband. `stretch_cap=0.20` pitch-preserving, `music_bed_gain=0.6`, 50 ms afade per line, per-sentence RMS loudness match (±12 dB), duck pre-roll 0.25 s / tail 0.10 s, `vocals_bed_gain=0.12` between sentences, no-overlap ripple with `voice_max_compress=1.5` | same |
| **output** | `mkv_default` — ffmpeg mux, Armenian default audio, original retained, Armenian SRT default | same |

**One model lives outside the registry:** `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN),
hardcoded as `_MODEL` in `patmir/pipeline/streaming/embeddings.py`, used for cross-section
speaker identity in watch-along. Cosine threshold **0.12** — calibrated, not taken from
literature (same-speaker-across-emotions drops to 0.157; different speakers ≤0.06).

### Higgs emotion control

Emotion reaches TTS as an inline `<|emotion:..|>` tag on the prompt, mapped from the 5
emotion bins by `EMOTION_TAGS` — identical in both runtimes, asserted by
`test_config_profiles.RuntimeParityTest`. Current map, every entry an **ear-test result**
(2026-08-24, two A/B rounds on real lines against real pinned refs):

| bin | tag |
| --- | --- |
| `angry` | **none** |
| `happy` | **none** |
| `sad` | `helplessness` |
| `fearful` | `shame` |
| `neutral` | none |

So ~43 % of a typical clip's sentences are deliberately untagged; emotion for those bins
comes from the cloned reference clip alone.

The checkpoint declares **62 control tokens** and the model card documents them; we use
two. Available: **21 emotion** (elation, amusement, enthusiasm, determination, pride,
contentment, affection, relief, contemplation, confusion, surprise, awe, longing,
arousal, anger, fear, disgust, bitterness, sadness, shame, helplessness), **10 prosody**
(`speed_very_slow/slow/fast/very_fast`, `pitch_low/high`, `pause`, `long_pause`,
`expressive_high`, `expressive_low`), **3 style** (singing, shouting, whispering),
**9 sfx**, **2 env**. Nothing upstream documents what any tag does acoustically — the
model card lists names and there is no paper, so the map has to be chosen by ear.

### Measured baselines

`samples/sample_1.mkv` — 301.11 s, Russian, dialogue-dense, ~60 % speech, 4 speakers,
~110 diarization turns. RTF = `wall_s / source_s`; **>1.0 = slower than realtime**.

| | batch (warm) | watch-along |
| --- | --- | --- |
| RTX 3090 / WSL2 | **0.461** (cold 0.958; builds = 53.4 % of a cold run) | **1.42** agg, **0.831 steady-state** |
| M1 Pro 32 GB | **1.17** | 1.611 |

CUDA batch stage split: **tts 55.3 %** (76.78 s, `tts_rtf` 0.309–0.347, 0 overflows),
asr 14.8 % (20.53 s), translate 13.6 % (18.82 s), separate 6.4 %, output 3.7 %,
mux 3.3 %, emotion 1.7 %, diarize 1.1 %, ingest 0.3 %. Peak 16.42 GB VRAM.

TTS is the largest stage on both machines. On CUDA, `tts_rtf` fell from MLX-q8's 1.136 to
~0.33 — synthesis is now faster than realtime per spoken second, which removes the floor
that kept batch above realtime on Apple Silicon.

---

## §5 Registered but not active

Revert paths and unscored options — **not** rejections. Present in `patmir.toml`, so
selectable by flipping one id.

**separate** — `htdemucs_mlx` (full `htdemucs_ft` 4-model bag), `htdemucs_mlx_fast`
(base model, used by watch-along on the Mac), `htdemucs` (torch, ft bag), `htdemucs_fast`
(torch base — active on the cuda profile's *streaming* layer).

**diarize** — `pyannote_3_1` (the id is a misnomer: its `model_id` is
`pyannote/speaker-diarization-community-1`), `elevenlabs_scribe_v2`.

**asr** — `whisper_large_v3_mlx_fp16` (`mlx-community/whisper-large-v3-mlx`),
`whisper_large_v3_turbo_mlx`, `nemotron_mlx`, `faster_whisper_large_v3` /
`_turbo` / `_fp16` (the last being the no-Qwen CUDA fallback), `qwen3_asr_0_6b`,
`qwen3_asr_1_7b_mlx_bf16` / `_4bit`, `qwen3_asr_0_6b_mlx_bf16`, `openrouter_gpt_audio`,
`elevenlabs_scribe_v2`.

**translate** — `nllb_200_3.3b` (the offline fallback), `nllb_200_distilled_600m`,
`openrouter_gemini_flash_batch` (batched + JSON-schema variant of the active candidate,
registered enabled but not selected), `qwen3_8b_mlx`
(`mlx-community/Qwen3-8B-4bit`), `qwen3_30b_a3b_gguf_q4km`
(`unsloth/Qwen3-30B-A3B-GGUF`).

**tts** — `higgs_v3_mlx_q8_fast`, `omnivoice_mlx_bf16` / `_bf16_fast` (the latter the
registered watch-along fallback) / `_q8` / `_q8_fast`, `piper_hye`
(`davit312/piper-TTS-Armenian`, `hy_AM-gor-medium` — the no-ref fallback and the EA
quality floor), `elevenlabs_multilingual_v2` (`eleven_v3`),
`openrouter_gemini_flash_tts`.

**Never evaluated, GPU-gated.** These sat in a "GPU re-eval queue" while the only machine
was a 32 GB Mac: `hygpt_10b_it` (`Gen2B/HyGPT-10b-it` — Gemma-2-9b base + 10B Armenian
tokens, ~20 GB VRAM fp16, described in the docs as likely the strongest local
translator), `madlad_400` (`google/madlad400-3b-mt`), `opus_mt_en_hy` /
`opus_mt_ru_hy` (`Helsinki-NLP`, ~80 MB each, EA explicitly tagged), and `suno/bark` for
TTS.

### One unresolved status

`qwen3_asr_1_7b_mlx_8bit` — the **active Mac ASR** — is recorded as **author-rejected on
quality on 2026-07-14** ("a lot of nonsense" to a native speaker) and was flipped active
**the same day** on the verdict "both garble sometimes, so take the fastest". The
confirming ear-test (`runs/20260715-reacher-asr-ab/`) is still pending. Its quality
standing is genuinely open, and this is a fact about the candidate rather than doc drift.

Related: the standalone A/B predicted a ~3 % in-pipeline edge over Whisper; the
2026-08-23 re-baseline found the edge much larger (ASR 10.8 % of batch wall vs Whisper's
29 %).

---

## §6 Rejected models, by stage

The payload. Format: **model — exact checkpoint — status + date — reason.**

A model here has been evaluated. Re-opening one needs new evidence (a new checkpoint, a
fixed upstream bug, a changed constraint), not a fresh opinion.

### separate

| Model | Checkpoint | Status | Reason |
| --- | --- | --- | --- |
| htdemucs_ft full 4-model bag | `htdemucs_ft` (sub-models `f7e0c4bc`, `d12395a8`, `92cfc3b6`, `04573f0d`) via `demucs-mlx` | superseded 2026-06-19, registered fallback | The bag's vocals output comes **100 %** from sub-model `04573f0d` — identity weight matrix, ear-confirmed corr 0.998 — so running only that sub-model is mathematically identical vocals at ~3.6× less stage time (39 → 11 s per 60 s), dropping the pipeline to ~1.07× realtime. |
| PyTorch Demucs on MPS | `demucs` PyPI, `htdemucs_ft` | registered fallback since 2026-06-15 | MLX port is faster on the author's Mac; torch path kept for CUDA and as a revert route. |

Further separate optimization is explicitly **not recommended** — already vocals-only,
~6 % of CUDA batch wall.

### diarize

| Model | Checkpoint | Status | Reason |
| --- | --- | --- | --- |
| pyannote 3.1 | `pyannote/speaker-diarization-3.1` | superseded 2026-05-27 | Buckets short turns from **distinct characters** into one `SPEAKER_NN` on fast cartoon dialogue. Forcing `num_speakers=4` did **not** fix it — the cluster count was right but sub-2 s embeddings were still confused. |
| pyannote community-1 | `pyannote/speaker-diarization-community-1` | registered fallback; **re-rejected 2026-08-23** as the CUDA pick | Still merged a 1.0 s female line into surrounding male turns. The first cut of `[pipeline.profiles.cuda]` selected it and was replaced by `sortformer_nemo` because it reintroduced quality issue #13: two characters share one pinned ref clip → one voice. |
| speechbrain ECAPA-TDNN diarizer | — | rejected 2026-05-26 | Open licence, but lower accuracy and **no native overlap detection**. |
| WhisperX bundled diarizer | — | rejected 2026-05-26 | Wraps pyannote; no independent advantage. |
| ElevenLabs Scribe v2 | `scribe_v2` | accepted 2026-05-27, superseded 2026-06-15; still registered | Credits exhausted. Also emitted **no overlap intervals** (`overlaps=[]`, quality issue #5) and no `transcribe_slice`, so the gap-filler silently skipped. |

Speaker-identity backends (watch-along, diarize-adjacent):

| Model | Status | Reason |
| --- | --- | --- |
| ReDimNet-b2 (`ptn` + `ft_lm`) | evaluated 2026-07-14, not adopted | Same/different-speaker cosine distributions **overlap** on short Demucs-separated emotional clips (ReDimNet −0.075 < +0.112; ECAPA +0.088 < +0.143), so per-clip matching is marginal either way. Left behind experimental flag `PATMIR_EMBED_BACKEND=redimnet` (threshold 0.02); ECAPA stays default. |
| Literature ECAPA threshold ~0.25–0.40 | rejected 2026-06-13 | Calibrated to **0.12** on real data — same-speaker across emotions drops to 0.157, different speakers stay ≤0.06. |

### emotion

| Model / approach | Status | Reason |
| --- | --- | --- |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | considered, skipped 2026-05-26 | Outputs dimensional valence/arousal/dominance — hard to map onto discrete ref-clip bins. |
| `superb/wav2vec2-base-superb-er` | skipped 2026-05-26 | Only 4 classes (ang/hap/neu/sad), IEMOCAP-only, less expressive than emotion2vec. |
| Emotion inferred from **translated text** | rejected 2026-05-26 (ADR) | Emotion must come from the source vocal, which carries ground-truth paralinguistic cues; translation rephrasing must not redefine it. |
| ModelScope as the emotion2vec download hub | rejected 2026-05-26 | ~500–700 KB/s from the US — 1.29 GB took 43 min. Candidate hardcodes `hub="hf"`. Generalized into the "prefer fast mirrors" ADR. |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` age head | partially rejected 2026-06-13 | Gender and child detection reliable; **age regression skews high**, so age is treated as approximate with conservative thresholds. |
| Per-`(speaker, emotion)` ref-clip binning as the emotion mechanism | effectively rejected 2026-06-19 | Caused characters' voices to swap mid-scene — `SPEAKER_00` cloned from 4 different short/noisy snippets (screams, laughs) — plus over-emoting. `tts_pin_speaker_ref=true` is now default (one ref per speaker, emotion ignored for ref choice). Binning survives behind `tts_pin_speaker_ref=false`. |
| The `[emotion_guidance]` table | **removed 2026-08-24** | Had been **inert since 2026-06-20**: the TTS adapter only forwards `guidance_scale` to candidates whose `synthesize` accepts it, and the Higgs candidates do not. Five values sat in `patmir.toml` looking load-bearing for two months. |

### asr

| Model | Checkpoint | Status | Reason |
| --- | --- | --- | --- |
| Nemotron 3.5 streaming | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | active 2026-06-15 → rejected 2026-06-19; registered as a fast alternate | **Garbles fast cartoon Russian** — Devanagari script leakage (`कंसर्ट`, `ना`) plus mangled words — while clean on slower Russian. Also emits **no punctuation**, forcing candidate-side re-segmentation. |
| Whisper large-v3-**turbo** (MLX) | `mlx-community/whisper-large-v3-turbo` | ear-rejected 2026-06-19 | The 8×-pruned text decoder (4 layers vs 32) produced **gibberish** and captured only ~half the words (173 vs 336). ~1.7× faster for ~1/3 of the dialogue lost. |
| faster-whisper large-v3 int8 | `large-v3`, `compute_type="int8"` | was Phase 1 default, superseded 2026-05-25 | CTranslate2 has **no MPS backend** → CPU-only on the Mac. |
| Qwen3-ASR-1.7B bf16 (MLX) | `mlx-community/Qwen3-ASR-1.7B-bf16` | measured 2026-07-15, no reason to use | Transcripts **character-for-character identical** to 8-bit on two clips, 2–4× slower (RTF 0.32–0.56 vs 0.10–0.15), +1.1 GB. **The errors are model-level, not quantization** — so the lever is a bigger checkpoint, and none above 1.7B existed on mlx-community as of that date. |
| Qwen3-ASR-1.7B 4bit (MLX) | `mlx-community/Qwen3-ASR-1.7B-4bit` | registered, not chosen | Pure 4-bit risks text-quality regressions on low-resource languages. |
| Voxtral Mini 4B Realtime | `mistralai/Voxtral-Mini-4B-Realtime-2602`, evaluated as `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | evaluated 2026-07-14, not adopted | Quality ≈ Whisper large-v3 (even rendered "Бонжорно" cleanly, no Nemotron-style garbling), but **RTF 0.64 vs Whisper's 0.40** and ~13× slower than Qwen3-ASR. Its causal-streaming design point is per-token latency for live streaming, not batch throughput over slices — it only pays off if watch-along moves to true sample-streaming ASR. Input gotcha: works only on **16 kHz mono**; produced `"."` on a 44.1 kHz stereo stem. |
| `openai/gpt-audio` (OpenRouter) | — | active 2026-05-26 morning, reverted the same evening; superseded 2026-05-27 | Cloud was not materially better than local Whisper and occasionally truncated the first sentence. |
| `openai/gpt-audio-mini` | — | disabled | **Safety-refuses cartoon clips** ("Извините, я не могу помочь с этим"). |
| `openai/gpt-4o-audio-preview` | — | disabled tripwire | Returns 404 `model_not_found` despite appearing in the OpenRouter catalog. |
| Voxtral Small 24B | `mistralai/voxtral-small-24b-2507` | disabled | Works, but swaps phonetically similar words (`Кислятина` → `Числятина`). ~$0.36/hr. |
| ElevenLabs Scribe v2 | `scribe_v2` | registered, disabled **by project policy** | Actually available on the Free tier — the gate is policy (cloud stays off until ear-tested and budget-sanctioned), not tier. |
| SeamlessM4T v2 Western-Armenian ASR | `ArthurMalajyan/seamless-m4t-v2-large-asr-hyw` | rejected | Western Armenian only — and the target language is wrong anyway: Patmir transcribes the **source**, not Armenian. |
| Whisper turbo Armenian FT | `Chillarmo/whisper-large-v3-turbo-armenian` | excluded | Armenian-**input** ASR. Wrong task. |

### translate

| Model | Checkpoint | Status | Reason |
| --- | --- | --- | --- |
| NLLB-200 distilled 600M | `facebook/nllb-200-distilled-600M` | superseded 2026-05-25, kept for legacy | Phase 0 scored idioms 1.33 and RU narration 2.10/2.20 — unusable. |
| NLLB-200-3.3B | `facebook/nllb-200-3.3B` | superseded as active 2026-05-26; **the offline fallback** | Register-neutral Armenian phrasing with **no prompt surface**, so no emotional dynamism for TTS to ride on. Was the secondary contributor to the flat ear-test that drove the cloud escalation. |
| NLLB-3.3B GGUF | `acceldium/nllb-200-3.3B-GGUF` (`nllb-3.3B.gguf`) | **dead end** | Requires a **fork** of llama.cpp implementing the `nllb` architecture; upstream does not, and `llama-cpp-python` builds against upstream. |
| TranslateGemma 27B / 12B / 4B | `mlx-community/translategemma-{27b,12b,4b}-it-4bit` | rejected 2026-05-27 | Two reasons. (1) The structured chat template accepts only `{type, source_lang_code, target_lang_code, text}` — **no system prompt**, so the casual-Yerevan register instruction and per-sentence emotion hint cannot be injected. Output skews literary (`Ցանկանո՞ւմ ես փորձել` where Gemini gives `Ուզո՞ւմ ես փորձես`) and it occasionally appends parenthesized Latin transliterations that cannot be suppressed. (2) ~7 tok/s on 27B-4bit MPS → **37 s/sentence** at `max_tokens=256`; a tuned version lands ~6 s, still 5–6× Gemini. Also `translategemma-27b-it-8bit` does not fit 32 GB, and mlx-lm 0.31.x does not apply the template at all (ml-explore/mlx-lm#783), so the candidate hand-builds the prompt string. |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | rejected 2026-05-27 | **4/10 sentences leaked chain-of-thought into the `content` field** even with `thinking={type:"disabled"}` (the model ignores OpenRouter's `reasoning.effort=minimal`). Average latency 10.9 s/sentence, ~10× Gemini 3.5 Flash. Revisit only if DeepSeek fixes the no-thinking-mode content leak. |
| Claude Opus 4.7 | `anthropic/claude-opus-4.7` | **user-rejected on cost** 2026-05-26 | Strongest on register and nuance, most expensive. **Do not re-enable without an explicit ask.** |
| GPT-5 | `openai/gpt-5` | registered, reliability concern | A/B showed casual register but **1/16 None-content errors** (reasoning model burning tokens internally). |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | superseded 2026-05-26 evening | Tilted formal/textbook even after the casual-register prompt rewrite — output read as "academic". |
| Gemini 3.1 Flash Lite | `google/gemini-3.1-flash-lite` | lost the 2026-05-26 A/B | 3.5-flash produced more emotive, playful Armenian (diminutives, exclamatory forms, casual vocatives) at the same speed. |
| Direct-vendor cloud | `gpt_cloud` (`gpt-4o`), `claude_cloud` (`claude-opus-4-7`) | disabled | Local-first-strict policy. Activation requires a new ADR naming local-exhaustion evidence. |
| Qwen3-14B GGUF | `unsloth/Qwen3-14B-GGUF` (`Qwen3-14B-Q4_K_M.gguf`) | disabled | **Visibly weaker on Armenian** than the 27B and 30B-A3B options; kept only as a small-memory fallback. |
| Qwen3 local variants (unscored) | `mlx-community/Qwen3-4B-Instruct-2507-4bit`, `Qwen3-14B-4bit`, `Qwen3.6-27B-4bit`, `Qwen3.6-27B-5bit`, `unsloth/Qwen3.6-27B-UD-MLX-4bit` | disabled | Registered, never ear-judged. Qwen3-Instruct-MLX as a whole still awaits an ear-test vs NLLB-3.3B; it needed reasoning suppression because it interleaved chain-of-thought with translations. |
| `reasoning enabled=false` on Gemini 3.5+ | — | **not possible** | HTTP 400. `effort="minimal"` is the workaround: reasoning_tokens 0, ~2 s vs ~4 s per sentence, ~30× cheaper. |

### tts

The most-explored stage and the most constrained — the EA + cloning + emotion triple
(§2.4) is what most of these fail.

**Rejected on language / dialect:**

| Model | Checkpoint | Status | Reason |
| --- | --- | --- | --- |
| MMS-TTS Eastern Armenian | `facebook/mms-tts-hye` | **does not exist** — 404 verified 2026-05-24 | Only the Western `mms-tts-hyw` is published. A stub candidate file is kept so future scans do not re-attempt the discovery. |
| MMS-TTS Western Armenian | `facebook/mms-tts-hyw` | disabled | Wrong dialect. Kept only so the author can hear a wrong-dialect baseline. |
| Coqui XTTS-v2 | `coqui/XTTS-v2` | rejected 2026-05-24 | The model card lists 17 languages and **Armenian is not one of them**. Earlier project notes claiming support were wrong; the entry is kept disabled with an inline correction so it is not re-added. |
| F5-TTS / F5-TTS-MLX | `SWivid/F5-TTS`, `lucasnewman/f5-tts-mlx` | rejected 2026-05-26 (research only) | EN + ZH base. Community fine-tunes exist for Arabic, Finnish, Spanish, French, German — **no EA fine-tune**. Building one needs an EA tokenizer plus ~50–100 h of EA speech: a project of its own. Otherwise attractive (335M, first-class MLX port, flow-matching DiT). |
| IndexTTS-2 | `IndexTeam/IndexTTS-2` | rejected 2026-05-26 | The **best emotion-control story in any open TTS** — explicit 8-dim emotion vector (happy/angry/sad/afraid/disgusted/melancholic/surprised/calm, each 0–1) decoupled from speaker identity. But ZH/EN/JA/ES only, **no MLX port**, 16 GB RAM minimum, no published quantized variants. The emotion vector is useless without the language coverage. |
| MeloTTS / OpenVoice v2 / Parler-TTS / SeamlessM4T | — | rejected on language | None list Armenian. |
| Bark | `suno/bark` | deferred indefinitely 2026-05-24 | Armenian dialect handling unverified; Bark sometimes produces unintelligible output on lower-resource languages. Eval cost was not worth a Phase 1 slip. |

**Rejected on quality or capability:**

| Model | Checkpoint | Status | Reason |
| --- | --- | --- | --- |
| Google Cloud TTS Armenian | `hy-AM-Standard-A` | disabled | Roughly **Piper-tier** quality — not a meaningful upgrade. (Azure `hy-AM-AnahitNeural` / `hy-AM-HaykNeural` are noted as a stronger cloud secondary but were never registered.) |
| Piper Eastern Armenian | `davit312/piper-TTS-Armenian`, `hy_AM-gor-medium` | demoted to fallback 2026-05-25; **specifically tried for watch-along and rejected** on quality | ~63 MB ONNX, CPU-only, trivially fast, native EA — but one fixed voice and no emotion control. Remains the **no-ref fallback** and the EA quality floor. |
| OmniVoice PyTorch fp16 | `k2-fsa/OmniVoice` | disabled | Heavier (~3 GB vs ~1 GB); cross-platform alternative only. |
| OmniVoice 8-bit / 4-bit (MLX) | `mlx-community/OmniVoice-8bit`, `-4bit` | rejected 2026-06-15 | `config.json` is missing the `quantization` block, so the model is not actually quantized and mlx-audio rejects the `.scales` weights. The card targets a **different runtime** (`ailuntx/OmniVoice-MLX`) and states its slim tokenizer does not provide a reliable speaker-encoding path — i.e. **no reliable cloning**. |
| OmniVoice bf16 (short name) | `mlx-community/OmniVoice-bf16` | rejected 2026-06-15 | Loads, but synthesis raises "tokenizer (HiggsAudioTokenizer) is required for voice cloning via ref_audio" — the repo's audio-tokenizer assets are not wired by mlx-audio's non-quantized load path. **No cloning.** |
| `ailuntx/OmniVoice-MLX` runtime | — | not adopted | Loads and clones, but **no faster** (fp16 RTF 0.97 @16; q4 1.07 @16 / 1.26 @32) — precision is not the speed lever, `num_steps` is. **q4@16 was ear-rejected.** |
| OmniVoice q8 g64 | `lightsofapollo/omnivoice-mlx-q8-g64` | demoted to fallback 2026-06-16 | Lost to bf16 on **both** axes. Ear-rank: bf16@32 EXCELLENT > q8@32 > bf16@16 (slight nasal) > q8@16 (inter-word pauses). Speed: bf16 RTF 0.46/0.89 @16/32 vs q8 0.67/1.20; load 1.7 s vs 6.1 s. |
| OmniVoice bf16 | `mlx-community/OmniVoice-bfloat16` | superseded 2026-06-20; registered fallback | **No explicit emotion knob.** Emotion had to route through a `guidance_scale` proxy plus per-emotion ref binning, which swapped voices mid-character and over-emoted (quality issues #3, #11). |
| OmniVoice `num_steps=8` | — | rejected on quality | Too low. 16 judged watchable, 32 good. |
| OmniVoice GGUF via `omnivoice.cpp` | `Serveurperso/OmniVoice-GGUF` (BF16 / Q8_0 / Q4_K_M), runtime `ServeurpersoCom/omnivoice.cpp` | rejected 2026-06-17 | Internal RTF ~1.89–2.04 across **all three quants** vs MLX bf16's ~1.1–1.5 (~2× slower), and quant level barely moves speed — compute-bound on Metal. **MLX bf16 won the ear-test** against all three. Adds a C++/Metal build, a ggml submodule and ~3.2 GB of GGUFs for no gain. `mg_num_step=32` is hardcoded with no CLI flag. |
| ElevenLabs pre-v3 models | `eleven_multilingual_v2`, `eleven_turbo_v2_5`, `eleven_flash_v2_5` | rejected 2026-05-26 | All return 400 `invalid_parameters` on `language_code="hy"`. Without the code they silently read Armenian script with English phonemes — "an English speaker doing fake Armenian". **Only `eleven_v3` accepts `hy`.** |
| ElevenLabs `eleven_v3` + IVC | `eleven_v3` | abandoned twice; registered | 2026-05-26: blocked on Free tier (library voices 402, IVC needs Creator ~$22/mo). 2026-05-27 PM: **ElevenLabs' captcha system blocked cloned voices of real celebrities** on YouTube content — HTTP 403 `voice_access_denied` / `detected_captcha_voice`. 2026-06-15: credits exhausted. |
| ElevenLabs preset-voice pool | `elevenlabs_pool` (`eleven_v3`, `clone=false`) | replaced 2026-06-15 | Was the watch-along TTS; dropped when credits ran out. Also the cost incident: an unbounded scheduler burned ~30K credits in a day on a 34-minute video. |
| OpenRouter Gemini Flash TTS | `google/gemini-3.1-flash-tts-preview` | active 2026-05-26 → same-day cloud alternate | Superseded once ElevenLabs IVC scope was added. Its voices are **not clones** (character identity lost, accepted at the time), and its emotion-tag granularity is coarser than ElevenLabs `voice_settings`. Ear verdict: "much better than what we had, not as good as ElevenLabs." |
| GPT-4o mini TTS | `openai/gpt-4o-mini-tts-2025-12-15` | disabled | Cheapest ($0.60/1M chars) but 6 fixed voices and **no audio tags** — no emotion control. |
| Voxtral Mini TTS | `mistralai/voxtral-mini-tts-2603` | disabled | Registered A/B alternate, never evaluated. |
| Surveyed, never registered (2026-05-26 OpenRouter sweep) | `hexgrad/kokoro-82m`, `x-ai/grok-voice-tts-1.0`, `zyphra/zonos-v0.1-transformer`, `zyphra/zonos-v0.1-hybrid`, `sesame/csm-1b`, `canopylabs/orpheus-3b-0.1-ft` | not registered | Lost to Gemini's 200+ inline audio tags at the time of the survey. |

**Higgs-specific rejections** (the active model — these are internal variants, not
alternatives):

| Variant / setting | Status | Reason |
| --- | --- | --- |
| Higgs q4 (our MLX port) | rejected 2026-06-20 | **Collapses** — runaway generation, never emits EOC. Consistent with Higgs being quant-sensitive. |
| Higgs bf16 on MLX | cross-platform fallback only | ~1.9 RTF, 2.5× slower than q8, with no quality need. Per-step: torch-bf16 69 ms → MLX-bf16 48 → MLX-q8 28. |
| mlx-audio's native `higgs_audio_v3` runtime (PR #770, closing #766) | no swap 2026-07-14 | Native-q8 was only **parity** (RTF 1.018 vs our 1.052), and native-q4 — 29 % faster at 0.744 with no collapse — was **author ear-rejected as the worst of the three**. "No collapse" is not the same as good quality. |
| `Reza2kn/Higgs-Audio-v3-TTS-4bit-MLX` | does not load | Different weight layout (`backbone.*` naming). Quantize in-memory from the base checkpoint instead. |
| Naive full-model quantization | rejected / reproduced-broken | Breaks the codec's VQ math. Only an **LM-only** quant predicate (codec and codebooks excluded) works. |
| `higgs_v3_mlx_q8_fast` | retired 2026-08-23 | Differed from `higgs_v3_mlx_q8` **only** in `max_retries` (1 vs 2) — identical repo, quant, temperature, top_p, seed. It was never a faster model: the cap only bounded the reseed path, and 0 overflows were measured across 25 + 37 + 60 sentences. One id lets a single 4B instance serve both modes from the warm cache, required on a 24 GB GPU. |
| Original emotion tag map | **ear-rejected 2026-08-24** | happy→`elation`, sad→`sadness`, angry→`anger`, fearful→`fear` — five obvious name-matches nobody had listened to. Across two A/B rounds (5×5 and 5×4 real lines on real pinned refs) **the shipped mapping was never preferred on a single line, in any bin**. Replaced by the §4 map. Pinned by tests that reject regression to those four. |
| `<\|prosody:expressive_low\|>` | real but not adopted 2026-08-24 | Demonstrably changes delivery (identical prompt ± the token gives different audio, md5-confirmed), but it won only 1 of 9 lines and on another made the line **more** emotional — the opposite of its name. Recorded as a caution: these tag names are labels, not descriptions. |
| Low temperature (cited 0.01–0.3 stability advice) | rejected — measured backwards | **LOW temperature causes overflow, not high.** At temp 0.3 a long line overflowed 5/5; 0.7–0.9 was 0/5. Low temp → near-greedy → repetition lock → never samples EOC. The generic advice targets long-paragraph serving. Defaults are temp 0.8 / top_p 0.95. |
| Higgs "scene description" system prompt (`<\|scene_desc_start\|>…`) | not used | The `generate_speech`-style path does not expose it. A possible untried quality lever. |
| `generate_speech()` as a thin wrapper (torch candidate) | abandoned 2026-08-23 | It is **batch-1** — the wrong shape for a bandwidth-bound decode — and there is no way to reach batching through upstream's public API. The decode loop now lives in `higgs_v3_torch.py` (~150 lines). |

### mux / output / ingest

| Approach | Status | Reason |
| --- | --- | --- |
| Pure-ripple overlap correction | rejected 2026-06-15 | Let drift reach ~11 s on dense sections. Replaced by a hybrid: compress the line pitch-preservingly (capped `voice_max_compress=1.5×`), then ripple the remainder. |
| A dedicated `yt_dlp` ingest candidate | deferred | yt-dlp runs as a pre-download in the `/jobs` handler instead; the `ffmpeg` ingest candidate is unchanged. |
| MP4 output candidate | designed, not built | Sibling module to `mkv_default`. |

---

## §7 Rejected approaches, runtimes and infrastructure

Not models, and equally important — these are the measurements that stop a research pass
re-deriving a conclusion.

### Quantization and precision

- **Higgs MLX q4** — collapses (see §6). **Higgs bf16** — 2.5× slower, fallback only.
- **mlx-audio native Higgs q4** — 29 % faster, ear-rejected.
- **Qwen3-ASR bf16** — strictly worse than 8-bit: character-identical transcripts, 2–4×
  slower. **The remaining ASR errors are model-level, not quantization**, so the lever is
  a bigger checkpoint — and none above 1.7B existed on mlx-community as of 2026-07-15.
- **OmniVoice q8 (g64)** — lost to bf16 on *both* quality and speed.
- **omnivoice.cpp Q4_K_M** — buys **no** speed over BF16; compute-bound on Metal.
- **Full-model (non-LM-only) Higgs quantize** — breaks codec VQ.

The general shape: for a memory-bandwidth-bound autoregressive decode, quantization is
the only speed lever (which is why Higgs is q8 on MLX). For an encoder-decoder ASR at
1.7B it buys nothing that matters.

### Runtimes and ports

- **GGUF + llama.cpp for NLLB** — dead end, no upstream `nllb` architecture.
- **GGUF + llama.cpp Metal for Qwen3-14B** — disabled, weaker Armenian than 27B / 30B-A3B.
- **`omnivoice.cpp`** (C++17 / GGML + Metal) — slower, lost the ear-test, adds a build dep.
- **The `qwen-asr` package — explicitly forbidden.** Pins `transformers==4.57.6` exactly,
  mutually exclusive with the Higgs checkpoint's `>=5.5`; installing it silently
  downgraded transformers and broke a working TTS stage. It also drags in exact pins on
  accelerate, nagisa and soynlp plus gradio/flask/sox. Use transformers-native
  `qwen3_asr` (5.13.0+). Note the **`-hf` suffix is load-bearing**: the unsuffixed repos
  (`Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ForcedAligner-0.6B`) are the `qwen-asr` layout and
  will not load through transformers.
- **`torch.compile` + torch micro-optimization on the Higgs decode — measured 0 %.**
  The decode is memory-bandwidth-bound; neither reduces bytes per token.
- **`flash_attention_2` for Higgs — refused by transformers.** The upstream port declares
  `_supports_sdpa` only, so `attn_implementation` must stay `sdpa` until upstream flags
  support. (The plan for the CUDA port had assumed flash-attention would be available; it
  was not.) For Qwen3-ASR the model card advertises it but it is **untried**.
- **`torchcodec` / `torchaudio.load` for reference clips** — abandoned 2026-08-24 after a
  `libtorchcodec_image.so` dlopen failure on the 3090 box. The obvious fallback —
  soundfile + `np.interp` resampling, as used in the separate stage's Silero frontend —
  was **rejected for ref clips specifically**, because linear interpolation aliases the
  audio a voice is cloned from. Ref clips now decode via ffmpeg
  (`eval/candidates/tts/_audio_io.py`).
- **NeMo install via git checkout + Cython prerequisite** — wrong; `nemo_toolkit 3.0.0`
  installs cleanly from PyPI. (The ADR and the candidate docstring both still claim
  otherwise — see [doc-drift.md](doc-drift.md).)

### Alignment and segmentation

- **MMS-FA forced alignment** (`ctc-forced-aligner`, ~1.2 GB `mms_fa.onnx`) — mandatory
  for cloud ASR from 2026-05-26, abandoned 2026-05-27 with the move to native word
  timestamps. It had a specific failure mode: long screaming/laughing sections
  transcribed as hundreds of repeated tokens dominated the alignment and crushed dialogue
  into tiny windows, and it padded durations (an 18.9 s single word).
- **"One `Sentence` per ASR chunk"** — a **quality bug**, not a design choice, fixed
  2026-08-24. Both Qwen3 candidates appended one `Sentence` per ASR chunk and never
  called `group_into_sentences`, so from 2026-07-14 onward "sentence" silently meant "up
  to 28 s of audio" **on both machines**: 23 sentences against diarization's 108 turns on
  a 5-minute clip, zero of 398 words carrying terminal punctuation (so nothing downstream
  could re-split), spans to 26.3 s. Because `apply_speaker_tags` assigns one *dominant*
  speaker per sentence, one 22.9 s span covering 11 turns at 86 % dominance was
  synthesized in **one character's voice** and swallowed another character's line. Fixed
  to 76–81 sentences with identical word count.
- **`max_gap_s` alone as a splitter** — insufficient. That 22.9 s run's widest interior
  gap was 0.32 s across 56 words, so a gap threshold could never have split it. Hence the
  `max_sentence_s` cap (8 s on the Qwen3 candidates, off elsewhere).
- **`_boundary_time` measured against the *dominant* speaker** — rejected. It silently
  failed whenever the dominant speaker was the **later** of two, which is half of all
  cases. Now measured against the speaker who *opens* the sentence.
- **VAD-snapped chunking as the finer-sentence mechanism** — hypothesis **measured wrong**.
  It gave 23 sentences against the MLX path's 25; chunking was never the mechanism.

### Concurrency, scheduling and performance

- **Thread-concurrency for MLX TTS — dead end.** MLX serializes on one GPU stream;
  concurrency 2 and 4 left wall-clock flat and only inflated per-sentence latency.
- **Section pipelining on MLX — not worth it.** Paper ceiling ~42 %, but both halves are
  GPU-bound, so only cloud-translate wait plus CPU ECAPA/remap actually overlap:
  **5–8 %, not 42 %**. Worth building on CUDA.
- **Per-speaker prefix-KV caching — dropped 2026-08-23.** Measured ceiling only −14.3 % of
  TTS time (decode is 73 % and is the real wall), and the CUDA move abandons the MLX port
  it would have optimized. Stated as a rule: do not tune code the active machine no
  longer runs.
- **Skipping `_free_accelerator_cache()` on warm cache hits — tried and reverted the same
  day (2026-08-23).** Watch-along aggregate RTF regressed 1.710 → **3.018**, every stage
  65–72 % slower — a uniform system-level stall. MLX Metal buffers live in unified memory
  largely *outside* RSS, so accumulating scratch was invisible to process accounting while
  driving the machine into swap. Clean re-measure under `caffeinate`: 1.611. Do not
  reintroduce the gate without a memory instrument that actually sees Metal allocations.
- **`bench_batch` with no `stage_cache`** — was deliberate, superseded 2026-08-24.
  Defensible when a build cost 2 s on MLX; not when it costs 154 s on torch (model builds
  were 53.4 % of a cold CUDA batch run). Cold RTF 0.958 → warm 0.461.
- **`prewarm_stage_cache` in bench** — deliberately not used: it would zero repeat 0's
  `build_s`, destroying the cold datapoint, and by contract it swallows build failures.
- **Persisting the MLX compile cache / cold-start kernel prewarm** (~240 s first run) —
  not recommended; fiddly for a once-per-session hit.
- **Benching without `caffeinate -i -m -s`** — rejected practice. A suspended run yields
  impossible per-stage rows (`time.time()` absorbs the suspend) beside a plausible
  aggregate.

### Architecture and configuration

- **D2 — stream per-sentence TTS WAVs as written, no cache** — rejected 2026-05-27. A
  half-step: still ~5–10 min before the first WAV, and no rewatch benefit.
- **D3 — pipeline-parallel runner + VAD chunking + parallel TTS** — rejected 2026-05-27,
  then **reopened 2026-06-13** when a throughput spike showed the premise was wrong. The
  rejection had assumed 0.64× realtime and TTS-bound with no fix short of a multi-week
  rewrite; the pipeline was actually 0.25× realtime with TTS at 74 % and `num_steps` a
  near-linear lever. Worth noting as a case where the *measurement*, not the reasoning,
  was the error.
- **DRM via system-audio capture** (macOS ScreenCaptureKit / BlackHole) — deferred
  indefinitely. EME-protected MSE buffers are opaque to extensions; system-audio capture
  is a different product.
- **"Chunk-everything" streaming** — rejected 2026-06-13 in favour of prelude + stream
  (marginally lower wait not worth the engineering plus early mis-assignment risk; fixed
  costs dominate). **Whole-file prelude** also rejected — wait scales with length.
- **Unbounded watch-along scheduler** (dub to completion for "free rewatch") — rejected
  2026-06-15 after ~30K ElevenLabs credits burned in a day. Replaced with bounded
  look-ahead (120 s), abandon timeout (45 s), and an explicit `POST /stream/<key>/stop`.
- **"Preset pool of 2–4 fixed EA voices" as the multi-voice plan** — scratched 2026-05-25;
  OmniVoice zero-shot cloning gave multi-voice for free.
- **Profile autodetection from `torch.cuda.is_available()`** — rejected 2026-08-23: which
  model ran has to be a stated choice or a benchmark stops being attributable. It also
  removes the ability to deliberately run a CPU/MLX set on the CUDA box.
- **Editing `[pipeline]` per-machine** — rejected; the one file deciding which model runs
  would diverge permanently across two working copies.
- **A second config file (`patmir.cuda.toml`)** — rejected; duplicates a ~700-line
  candidate registry and the copies drift.
- **"Unknown or unset profile is always a hard error"** — corrected 2026-08-23. Too
  aggressive: `$PATMIR_PROFILE` is machine-scoped, so it hit minimal test configs and
  broke 5 `test_paths.py` tests on the workstation only. Now keyed on provenance, with a
  `conftest.py` clearing the variable for the suite.
- **Storage-roots alternatives** (ADR 2026-08-01) — publish-destination-only (the author
  wanted every root movable); **env vars / per-run CLI flags** (author: one obvious place
  beats three precedence layers); flat `<stem>.hy.mkv` naming (repeat runs collide);
  **moving** rather than copying deliverables (breaks `--resume-from` and
  `cache.materialize`).
- **Non-OpenRouter cloud STT** — Soniox, Gladia, Deepgram, Speechmatics, and direct
  OpenAI `/v1/audio/transcriptions` are **off the table** on an author constraint: "use
  what's available on OpenRouter". A second key/account was not accepted despite richer
  native features.
- **OpenRouter `/v1/audio/transcriptions`** — **does not exist** (verified 2026-05-26;
  every transcribe-SKU id 500s). And `openai/gpt-4o-transcribe`, `openai/whisper-1`,
  `qwen/qwen3-asr-flash`, `google/chirp-3`, `mistralai/voxtral-mini-transcribe` are **not
  in OpenRouter's catalog** — 356 models enumerated, only 4 audio-capable, all
  chat-completions multimodal.

### Diagnoses that were wrong

Recorded so they are not re-derived.

**The RTX 3090 OOM.** `CUDA out of memory. Tried to allocate 20.00 MiB … 14.46 GiB is
free` was the **Windows commit limit**, not VRAM. CUDA under WSL2 charges host commit
~1.06 GB per GB of VRAM, so `commit limit = RAM + pagefile` caps GPU allocation; a 4 GB
pagefile left ~16 GB usable on a 24 GB card. Fixed by raising the pagefile to 20 GB.

Recognisable by shape: a 20 MiB request failing with 14 GiB free is never a real capacity
problem, and the reported `17179869184.00 GiB` of "non-PyTorch memory" is 2³⁴ GiB, i.e.
garbage — PyTorch infers it by subtraction and the arithmetic goes negative because the
refusal did not come from VRAM accounting at all. The traceback lands on whatever tiny
allocation came next, blaming innocent code, and the suggested
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not help because there is no
fragmented pool to compact.

Four explanations advanced and disproven along the way:

- `.wslconfig memory=24GB` starving Windows — it is a cap, not a reservation.
- The warm stage cache retaining GPU inference state — it holds 6.49 GB, exactly the
  weights.
- Fragmentation — `expandable_segments:True` changed nothing; reserved-but-unallocated
  was 15 MiB.
- Faulty VRAM (76 GiB written and verified, 0 errors) and the Windows "shared GPU memory"
  pool (CUDA under WSL2 does not use it).

**A red herring in the Higgs load report.** `audio_head.weight | MISSING | newly
initialized` is a **tied weight**, tied *after* the report prints. The candidate now
asserts the tie at build time so the message cannot mislead again.

---

## §8 Where each fact lives

| Question | File |
| --- | --- |
| What runs each stage (**canonical**) | `patmir.toml` — `[pipeline]`, `[pipeline.profiles.*]`, and the candidate registry |
| Stage contracts | `patmir/pipeline/stage.py` (protocols), `docs/architecture/data-contracts.md` (every field) |
| Profile resolution and namespaces | `patmir/pipeline/config.py` |
| Why a choice was made | `docs/architecture/decisions/` (25 ADRs), indexed in `decisions.md` |
| Per-stage model write-ups | `docs/architecture/stages/<stage>/` — `index.md`, `current.md`, one file per candidate |
| Performance baselines | `docs/architecture/performance.md` — **re-measure, do not re-derive** |
| Known output problems | `docs/architecture/quality-issues.md` (live tracker) |
| The Eastern-Armenian gate | `docs/architecture/language-support.md` |
| Which clip a number can be reproduced on | `docs/architecture/eval-clips.md` |
| Docs that disagree with `patmir.toml` | `docs/architecture/doc-drift.md` |
| CUDA/WSL2 environment gotchas | `docs/setup/windows-wsl2-cuda.md` |

**Eval clips.** `samples/sample_1.mkv` (301.113 s, **Russian**, dialogue-dense, 4
speakers, ~110 turns) is the only current dialogue clip and the one every measurement in
`performance.md` names. `samples/clip-en-test.mkv` (7.183 s, English) is a cross-language
sanity check only — too short to amortize model builds, so its RTF measures the
fixed-cost floor and nothing else. The historical `masha-2-5min.mkv` and
`Fallout-S02E01-5min.mkv` are **gone from disk**, so every pre-2026-08-24 number citing
them is unreproducible.
