# A translator outage must fail the dub, not mute it — Shaping Notes

## Scope

Bug report (author, 2026-09-15): *"the server was running, the browser plugin (latest
version) was showing that it translates the video and chunks are being translated
(1, 2, .... 5,...) but I was playing the video original (non translated, english) voice was
playing (a little bit muffled compared to the voice played when the translation was off).
please inspect the logs, it happened yesterday at evening."* Mid-shaping the author added:
*"openrouter credit went to 0, toped up the account."*

What this spec changes: a translator that cannot translate anything must end the stream or
job `failed` with a readable reason, the extension must hand the original audio back and show
the reason, and a section with no Armenian in it must never reach the cache.

## Log evidence (`~/.patmir/logs/server.log`, `runs/20260914-210424-section`)

- First `HTTP 402 Payment Required` from `openrouter.ai/api/v1/chat/completions`:
  **2026-09-13 09:49:01**. 1,546 402s on 09-13, 223 on 09-14. OpenRouter's credits endpoint
  at shaping time: `total_credits 20, total_usage 20.18`.
- 09-13 09:49-10:00 mixed 200/402 (partial-empty sections: "6 of 13", "4 of 11"); from
  09:59 every section 100% empty.
- Per 12-sentence section: 13 402s = 1 batch call + 12 per-sentence recoveries. Then
  `patmir.pipeline.stages.translate.adapter: 12 of 12 line(s) came back empty from the
  translator (audience=teenager) — those lines will be silent`. Every stage `done`.
- Section artifacts: `translated.json` 7 lines, all `target_text: ""`; `synthesized.json`
  0 sentences; `mux_placements.json` `[]`; `tts/voice_cloning_summary.json` `sentences: 0`;
  `section_dub.wav` mean -31.4 dB against `audio.wav` -14.6 dB and `accompaniment.wav`
  -32.2 dB, i.e. `0.6*accompaniment + 0.12*vocals`.
- Blast radius: 09-13 after the cutoff 209 runs, 1,421 of 2,492 sentences empty; 09-14
  17 runs, 206 of 206 empty.

## Poisoned cache (deleted 2026-09-15 per the author's decision)

Cutoff `produced_at >= 2026-09-13 09:49:01`. `tts=0` is `stage_wall_s.tts == 0.0`.

| key | sections | after cutoff | tts=0 | source |
| --- | --- | --- | --- | --- |
| `17fc1b081e03f729` | 71 | 25 | 1 (partial-empty window) | youtube `nrY-8_hJLJo` |
| `3f9ce3f2c3cfe9be` | 11 | 11 | 11 | youtube `QpwJEYGCngI` |
| `47db7fc315969105` | 29 | 29 | 29 | youtube `nBfK04-QPpg` |
| `5f7d4767ec45afbb` | 18 | 18 | 18 | youtube `ltJswDvMgY8` |
| `70be3ae34a039a08` | 7 | 7 | 7 | youtube `9IiYOTzJ2uw` (the reported evening) |
| `9b6cc113acc9b825` | 4 | 4 | 4 | youtube `LZxHd0_J2aQ` |
| `9bc1d6b1ec513e95` | 17 | 17 | 17 | youtube `oUIk1KLFXOY` |
| `c7e79dd6873c515b` | 22 | 22 | 22 | youtube `rSgvY33JRt4` |
| `c8e666c513e8fa71` | 4 | 4 | 4 | youtube `h5Zwg3Ag-bE` |

`efa962dcde154507` (18 sections, produced after the top-up, all with TTS work) was left alone.
Downloads under `~/.patmir/downloads/` were left alone (keyed audience-free, still valid).

## Decisions

- **Fail loudly** (author): typed `TranslatorUnavailable`, stream/job `failed` with a plain
  message, extension releases the video. No local auto-fallback.
- **Delete the nine poisoned cache dirs wholesale** (author), not surgical per-section removal.
- **No preflight credit probe** (author): the first section's failure is the signal.
- Refusal-vs-outage boundary: 401/402/404 from OpenRouter -> `TranslatorUnavailable`; 403
  (per-input moderation) and 200 with `content: null` stay per-line empties per the audience
  ADR; every line of N >= 2 empty at the adapter -> `TranslatorUnavailable` as a backstop; one
  line empty stays a reported gap.

## Incident during implementation

`uv run pytest` **removed and recreated `.venv`**: `.python-version` pins 3.11 while the
working venv on this box is Python 3.12 (per `docs/setup/windows-wsl2-cuda.md`), so uv treated
the environment as stale. The live server (started from `.venv/bin/patmir`) broke on its next
lazy import (`silero_vad.jit does not exist`). Rebuilt per the setup doc's Phase 7, pinned to
the versions still in the uv cache (torch 2.11.0+cu128, torchaudio 2.11.0+cu128,
transformers 5.16.1, nemo_toolkit 3.0.0), then restarted the server. Recorded as a gotcha in
the setup doc; on this box run tests with `.venv/bin/python -m pytest`, never `uv run`.

## Context

- **Visuals:** None.
- **References:** see `references.md`.
- **Product alignment:** local-first (cloud translate is the one opt-in cloud stage); the
  dub is the listener's only audio track, so a silent or voiceless dub is the worst outcome.
  Nothing here changes the model set or the cache key.

## Standards Applied

None — `agent-os/standards/index.yml` is empty.
