# Product Mission

*Patmir (պատմիր): tell the story, in his language.*

## Problem

Armenian-dubbed video content (movies, cartoons, YouTube, live streams) is scarce and largely unavailable through commercial channels. Children who are fluent in Armenian but struggle with foreign-language audio — including those with hearing aids, for whom learning a new language is especially difficult — are effectively locked out of most of the world's video content. Subtitles do not solve this: they require literacy in the foreign language, divide attention from the picture, and are inaccessible to pre-readers or to viewers whose hearing aids do not pair well with reading-heavy content.

## Target Users

**Primary (v1):** The author's son — fluent in **Eastern Armenian** as a mother tongue, wears hearing aids, finds learning new languages difficult. A child who simply wants to watch the same shows his peers do, in a language he understands.

**Designed for, post-v1:** Other Eastern-Armenian-speaking families in similar situations — particularly those with children, hearing-impaired family members, or elderly relatives — who want access to foreign content in Armenian. Patmir is a personal workstation first, but its architecture should not preclude others from running it.

### Dialect: Eastern Armenian only

This is a hard scope decision, not a default. Armenian splits into two literary standards:
- **Eastern Armenian** (`hy_AM`, ISO 639-3 `hye`) — spoken in Armenia and by the Iranian-Armenian community; the target dialect.
- **Western Armenian** (`hyw`) — spoken by most diaspora communities (Lebanon, Syria, US, France).

The two share the Armenian script but differ meaningfully in pronunciation (notably the consonant voicing system) and in vocabulary. A Western-Armenian TTS reading Eastern-Armenian text would be intelligible to a child but would sound foreign — which defeats the whole point of the product. So:

- Target locale is `hy_AM` end-to-end.
- Western-Armenian-only models are not viable candidates; if one is the strongest available, the product is not buildable today.

This is **not** a consumer SaaS product. It is a workstation tool. A polished UI is not a goal.

## Solution

A local-first translation workstation that takes foreign-language video (file, YouTube URL, or stream) and produces a watchable version with Armenian audio. The pipeline is modular — audio ingestion, ASR, translation, Armenian TTS, audio mux — and every stage is a swappable component with a clear interface, so individual models can be upgraded as the open-model ecosystem evolves.

**What makes Patmir distinctive:**

- **Eastern Armenian as a first-class target**, not an afterthought. Eastern Armenian is a lower-resource language for most modern translation and TTS models — and *Western* Armenian, where it's supported at all, is the wrong dialect for the target user. Treating Eastern-Armenian-specific model availability as the central technical risk shapes every architecture decision (early eval before stack commitment, swappable stages, willingness to fall back to cloud for one stage without compromising the others).
- **Local-first by default**, cloud only as a per-stage, opt-in fallback. The product runs on a single capable workstation with a GPU. Privacy, offline capability, and zero per-use cost are core values, not afterthoughts.
- **Modularity over cleverness.** A clean pipeline with swappable stages beats a tightly-coupled "smart" system, because the bottleneck stage (almost certainly Armenian TTS quality) will change as the model landscape changes.
- **Audio output is the product**, not subtitles. Subtitles may fall out as a byproduct, but the goal is for a child to *watch and listen*, not read.

## Non-Goals (v1)

- Lip sync or video re-rendering of mouths.
- Subtitle-only output as the primary feature.
- Multi-target-language output. Armenian only.
- A polished consumer UI or distribution as a product.
