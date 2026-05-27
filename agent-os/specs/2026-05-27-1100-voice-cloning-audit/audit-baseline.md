# Voice-cloning audit — `20260526-183305-masha-2-5min`

## Headline metrics

- Speakers detected: **5**
- Speakers actually cloned (entry in `refs.json`): **1**
- Speakers with a ref WAV on disk but no entry in `refs.json` (would be recovered by the fix): **4**
- Total sentences in `translated.json`: **36**
- Sentences by adapter fallback step:
    - `default_voice`: 29
    - `exact`: 4
    - `any_speaker`: 3
- `min_ref_seconds` assumed: 3.0

## Per speaker

| speaker | turns | clean s | longest eligible s | refs on disk | refs in JSON | status |
| --- | --- | --- | --- | --- | --- | --- |
| SPEAKER_00 | 86 | 12.88 | 4.68 | happy | — | `would_clone_after_fix` |
| SPEAKER_01 | 4 | 4.78 | 4.78 | sad | — | `would_clone_after_fix` |
| SPEAKER_02 | 18 | 5.37 | 5.37 | angry | — | `would_clone_after_fix` |
| SPEAKER_03 | 18 | 10.25 | 4.10 | angry, happy | happy | `cloned` |
| SPEAKER_04 | 13 | 3.01 | 3.01 | angry | — | `would_clone_after_fix` |

## Per-(speaker, emotion) ref detail

| speaker | emotion | wav | in refs.json | source window | backfill words | backfill text |
| --- | --- | --- | --- | --- | --- | --- |
| SPEAKER_00 | angry | — | — | — | 0 | (empty) |
| SPEAKER_00 | fearful | — | — | — | 0 | (empty) |
| SPEAKER_00 | happy | ✓ | — | 226.44–231.12 | 5 | -ха -ха! Ха -ха -ха! |
| SPEAKER_00 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_00 | sad | — | — | — | 0 | (empty) |
| SPEAKER_01 | happy | — | — | — | 0 | (empty) |
| SPEAKER_01 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_01 | sad | ✓ | — | 93.23–98.01 | 10 | -фью -фью -фью -фью -фью! Цыпа -цыпа -цыпа -цыпа -цыпа! |
| SPEAKER_02 | angry | ✓ | — | 150.40–155.77 | 11 | мы тут, э -это, прохлаждаемся. Ну, это, мороженое поели! Ну … |
| SPEAKER_02 | happy | — | — | — | 0 | (empty) |
| SPEAKER_02 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_02 | sad | — | — | — | 0 | (empty) |
| SPEAKER_03 | angry | ✓ | — | 30.07–33.21 | 8 | -ням -ням -ням -ням -ням! Эй, верни моё! |
| SPEAKER_03 | happy | ✓ | ✓ | 17.75–21.85 | 8 | попробовать? Давай ты моё возьмёшь, а я твоё! |
| SPEAKER_03 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_04 | angry | ✓ | — | 192.76–195.77 | 3 | Солисту горло посмотрите! |
| SPEAKER_04 | happy | — | — | — | 0 | (empty) |
| SPEAKER_04 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_04 | sad | — | — | — | 0 | (empty) |

## Per-sentence fallback path

| idx | speaker | emotion | step | matched ref |
| --- | --- | --- | --- | --- |
| 0 | SPEAKER_00 | angry | `default_voice` | — |
| 1 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 2 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 3 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 4 | SPEAKER_03 | angry | `any_speaker` | SPEAKER_03__happy |
| 5 | SPEAKER_03 | angry | `any_speaker` | SPEAKER_03__happy |
| 6 | SPEAKER_00 | angry | `default_voice` | — |
| 7 | SPEAKER_00 | sad | `default_voice` | — |
| 8 | SPEAKER_00 | neutral | `default_voice` | — |
| 9 | SPEAKER_00 | happy | `default_voice` | — |
| 10 | SPEAKER_00 | happy | `default_voice` | — |
| 11 | SPEAKER_02 | neutral | `default_voice` | — |
| 12 | SPEAKER_02 | angry | `default_voice` | — |
| 13 | SPEAKER_02 | happy | `default_voice` | — |
| 14 | SPEAKER_02 | sad | `default_voice` | — |
| 15 | SPEAKER_02 | angry | `default_voice` | — |
| 16 | SPEAKER_02 | angry | `default_voice` | — |
| 17 | SPEAKER_02 | angry | `default_voice` | — |
| 18 | SPEAKER_02 | happy | `default_voice` | — |
| 19 | SPEAKER_02 | happy | `default_voice` | — |
| 20 | SPEAKER_02 | happy | `default_voice` | — |
| 21 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 22 | SPEAKER_03 | angry | `any_speaker` | SPEAKER_03__happy |
| 23 | SPEAKER_04 | angry | `default_voice` | — |
| 24 | SPEAKER_04 | angry | `default_voice` | — |
| 25 | SPEAKER_00 | angry | `default_voice` | — |
| 26 | SPEAKER_00 | happy | `default_voice` | — |
| 27 | SPEAKER_02 | angry | `default_voice` | — |
| 28 | SPEAKER_02 | happy | `default_voice` | — |
| 29 | SPEAKER_00 | sad | `default_voice` | — |
| 30 | SPEAKER_04 | happy | `default_voice` | — |
| 31 | SPEAKER_04 | angry | `default_voice` | — |
| 32 | SPEAKER_04 | angry | `default_voice` | — |
| 33 | SPEAKER_04 | angry | `default_voice` | — |
| 34 | SPEAKER_04 | angry | `default_voice` | — |
| 35 | SPEAKER_00 | happy | `default_voice` | — |
