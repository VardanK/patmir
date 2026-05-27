# Voice-cloning audit — `20260526-183305-masha-2-5min`

## Headline metrics

- Speakers detected: **5**
- Speakers actually cloned (entry in `refs.json`): **5**
- Speakers with a ref WAV on disk but no entry in `refs.json` (would be recovered by the fix): **0**
- Total sentences in `translated.json`: **36**
- Sentences by adapter fallback step:
    - `exact`: 33
    - `speaker_neutral`: 2
    - `any_speaker`: 1
- `min_ref_seconds` assumed: 1.5

## Per speaker

| speaker | turns | clean s | longest eligible s | refs on disk | refs in JSON | status |
| --- | --- | --- | --- | --- | --- | --- |
| SPEAKER_00 | 86 | 27.33 | 4.68 | angry, fearful, happy, sad | angry, fearful, happy, sad | `cloned` |
| SPEAKER_01 | 4 | 7.51 | 4.78 | happy, sad | happy, sad | `cloned` |
| SPEAKER_02 | 18 | 15.78 | 5.37 | angry, happy, neutral | angry, happy, neutral | `cloned` |
| SPEAKER_03 | 18 | 11.89 | 4.10 | angry, happy | angry, happy | `cloned` |
| SPEAKER_04 | 13 | 10.86 | 3.01 | angry, neutral | angry, neutral | `cloned` |

## Per-(speaker, emotion) ref detail

| speaker | emotion | wav | in refs.json | source window | backfill words | backfill text |
| --- | --- | --- | --- | --- | --- | --- |
| SPEAKER_00 | angry | ✓ | ✓ | 214.51–216.23 | 5 | Ааа, -ха не качай! -ха! |
| SPEAKER_00 | fearful | ✓ | ✓ | 75.38–76.90 | 4 | -ха! Ха -ха -ха! |
| SPEAKER_00 | happy | ✓ | ✓ | 226.44–231.12 | 5 | -ха -ха! Ха -ха -ха! |
| SPEAKER_00 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_00 | sad | ✓ | ✓ | 60.27–61.93 | 4 | пожаловать в наш отель! |
| SPEAKER_01 | happy | ✓ | ✓ | 89.69–92.42 | 4 | Цыпа -цыпа -цыпа -цыпа! |
| SPEAKER_01 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_01 | sad | ✓ | ✓ | 93.23–98.01 | 10 | -фью -фью -фью -фью -фью! Цыпа -цыпа -цыпа -цыпа -цыпа! |
| SPEAKER_02 | angry | ✓ | ✓ | 150.40–155.77 | 11 | мы тут, э -это, прохлаждаемся. Ну, это, мороженое поели! Ну … |
| SPEAKER_02 | happy | ✓ | ✓ | 134.10–136.43 | 3 | Нам пора репетировать! |
| SPEAKER_02 | neutral | ✓ | ✓ | 116.89–118.56 | 2 | мама миа! |
| SPEAKER_02 | sad | — | — | — | 0 | (empty) |
| SPEAKER_03 | angry | ✓ | ✓ | 30.07–33.21 | 8 | -ням -ням -ням -ням -ням! Эй, верни моё! |
| SPEAKER_03 | happy | ✓ | ✓ | 17.75–21.85 | 8 | попробовать? Давай ты моё возьмёшь, а я твоё! |
| SPEAKER_03 | neutral | — | — | — | 0 | (empty) |
| SPEAKER_04 | angry | ✓ | ✓ | 192.76–195.77 | 3 | Солисту горло посмотрите! |
| SPEAKER_04 | happy | — | — | — | 0 | (empty) |
| SPEAKER_04 | neutral | ✓ | ✓ | 189.13–191.65 | 1 | О! |
| SPEAKER_04 | sad | — | — | — | 0 | (empty) |

## Per-sentence fallback path

| idx | speaker | emotion | step | matched ref |
| --- | --- | --- | --- | --- |
| 0 | SPEAKER_00 | angry | `exact` | SPEAKER_00__angry |
| 1 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 2 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 3 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 4 | SPEAKER_03 | angry | `exact` | SPEAKER_03__angry |
| 5 | SPEAKER_03 | angry | `exact` | SPEAKER_03__angry |
| 6 | SPEAKER_00 | angry | `exact` | SPEAKER_00__angry |
| 7 | SPEAKER_00 | sad | `exact` | SPEAKER_00__sad |
| 8 | SPEAKER_00 | neutral | `any_speaker` | SPEAKER_00__happy |
| 9 | SPEAKER_00 | happy | `exact` | SPEAKER_00__happy |
| 10 | SPEAKER_00 | happy | `exact` | SPEAKER_00__happy |
| 11 | SPEAKER_02 | neutral | `exact` | SPEAKER_02__neutral |
| 12 | SPEAKER_02 | angry | `exact` | SPEAKER_02__angry |
| 13 | SPEAKER_02 | happy | `exact` | SPEAKER_02__happy |
| 14 | SPEAKER_02 | sad | `speaker_neutral` | SPEAKER_02__neutral |
| 15 | SPEAKER_02 | angry | `exact` | SPEAKER_02__angry |
| 16 | SPEAKER_02 | angry | `exact` | SPEAKER_02__angry |
| 17 | SPEAKER_02 | angry | `exact` | SPEAKER_02__angry |
| 18 | SPEAKER_02 | happy | `exact` | SPEAKER_02__happy |
| 19 | SPEAKER_02 | happy | `exact` | SPEAKER_02__happy |
| 20 | SPEAKER_02 | happy | `exact` | SPEAKER_02__happy |
| 21 | SPEAKER_03 | happy | `exact` | SPEAKER_03__happy |
| 22 | SPEAKER_03 | angry | `exact` | SPEAKER_03__angry |
| 23 | SPEAKER_04 | angry | `exact` | SPEAKER_04__angry |
| 24 | SPEAKER_04 | angry | `exact` | SPEAKER_04__angry |
| 25 | SPEAKER_00 | angry | `exact` | SPEAKER_00__angry |
| 26 | SPEAKER_00 | happy | `exact` | SPEAKER_00__happy |
| 27 | SPEAKER_02 | angry | `exact` | SPEAKER_02__angry |
| 28 | SPEAKER_02 | happy | `exact` | SPEAKER_02__happy |
| 29 | SPEAKER_00 | sad | `exact` | SPEAKER_00__sad |
| 30 | SPEAKER_04 | happy | `speaker_neutral` | SPEAKER_04__neutral |
| 31 | SPEAKER_04 | angry | `exact` | SPEAKER_04__angry |
| 32 | SPEAKER_04 | angry | `exact` | SPEAKER_04__angry |
| 33 | SPEAKER_04 | angry | `exact` | SPEAKER_04__angry |
| 34 | SPEAKER_04 | angry | `exact` | SPEAKER_04__angry |
| 35 | SPEAKER_00 | happy | `exact` | SPEAKER_00__happy |
