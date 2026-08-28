"""URL-keyed global cache for finished dub jobs.

`runs/<job_id>/` is the ephemeral workspace the pipeline writes into.
`~/.patmir/cache/<key>/` is the stable, curated export of a finished job
keyed by `(source_id, pipeline_config)`. The Chrome extension hits
`/cache/lookup` on every YouTube tab load — if the active video's URL hashes
to a key that exists here, the popup shows "Play dub" instead of "Translate
this".

Key derivation deliberately excludes audio-tuning options (music_bed_gain,
voice_gain_db, sentence_afade_seconds) so ear-test tweaks don't bust the
cache — only stage selectors + per-candidate config dicts contribute.

The single exception is the audience age level, which changes the Armenian
TEXT rather than its mix. It participates via an explicit `cache_key(...)`
suffix rather than through `config_hash`, because `config_hash` hashes
candidate ENTRIES and the level is not a model selection — putting it there
would fragment the warm stage cache, which is keyed `(stage, candidate_id)`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from patmir import paths
from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import NEUTRAL_AUDIENCE
from patmir.pipeline.stage import STAGE_ORDER

log = logging.getLogger(__name__)

CACHE_VERSION = 1
# Fallback for callers with no loaded config. The server takes its cache root
# from `PipelineConfig.storage.cache_root` so `[storage]` is honored.
DEFAULT_CACHE_ROOT = paths.defaults(None).cache_root
_BY_URL_FILE = "by-url.json"


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    cache_dir: Path
    source_id: str
    source_url: str | None
    duration: float | None
    created_at: float
    job_id: str

    def dub_wav(self) -> Path:
        return self.cache_dir / "dub.wav"

    def metadata_path(self) -> Path:
        return self.cache_dir / "metadata.json"


def canonicalize_url(url: str) -> str:
    """Reduce a URL to a stable identifier across sessions / tracking params.

    YouTube watch/short/embed/youtu.be all collapse to `yt:<videoId>`.
    Anything else uses `scheme://host/path` (query + fragment dropped) — a
    conservative canonicalization that's stable across most CDN-fronted media.
    """
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host.endswith("youtube.com") or host == "youtu.be":
        video_id = _youtube_video_id(parsed)
        if video_id:
            return f"yt:{video_id}"
    # Generic fallback: scheme + host + path, no query / fragment.
    scheme = parsed.scheme or "https"
    path = parsed.path or "/"
    return f"{scheme}://{host}{path}".rstrip("/")


def _youtube_video_id(parsed) -> str | None:
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.lstrip("/").split("/", 1)[0] or None
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    # /shorts/<id> and /embed/<id>
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "v", "live"}:
        return parts[1]
    return None


def source_id_for_url(url: str) -> str:
    return f"url:{canonicalize_url(url)}"


def source_id_for_file(path: Path) -> str:
    """Fast content-hash for a local media file.

    `(size, sha256(first_1MiB + last_1MiB))` is fast on multi-GB MKVs and
    collision-resistant enough for a single-user local cache. Re-encodes
    will (intentionally) miss the cache.
    """
    path = path.resolve()
    size = path.stat().st_size
    chunk = 1 << 20  # 1 MiB
    h = hashlib.sha256()
    with path.open("rb") as f:
        head = f.read(chunk)
        h.update(head)
        if size > chunk:
            f.seek(max(0, size - chunk))
            tail = f.read(chunk)
            h.update(tail)
    return f"file:{size}:{h.hexdigest()[:16]}"


def config_hash(cfg: PipelineConfig) -> str:
    """8-hex digest of the load-bearing pipeline configuration.

    Hashes the candidate entry dict for each stage in `STAGE_ORDER`. The
    entry includes the candidate `id`, its `module`, and every model-id /
    voice / language knob the candidate exposes. Audio-tuning options under
    `[pipeline.options]` are deliberately excluded — and so is the audience
    level, which is a viewer preference rather than a model selection and rides
    the `cache_key` suffix instead (keeping the warm stage cache un-fragmented).
    """
    payload = {
        "version": CACHE_VERSION,
        "target_lang": "hy",
        "stages": {
            name: _stable_entry(cfg.entries.get(name, {})) for name in STAGE_ORDER
        },
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def _stable_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe + stable-ordered copy of a candidate entry."""
    # Sort keys; drop nothing — every knob in the candidate entry can change
    # output. TOML loads scalars + lists + sub-dicts, all JSON-serializable.
    return json.loads(json.dumps(entry, sort_keys=True, ensure_ascii=False))


# The default (D1) mode. Folded into the cache key only when non-default, so
# existing quality-mode cache entries keep their keys (no one-time cache bust)
# while watch-along caches under a distinct key — the two modes never collide.
DEFAULT_MODE = "quality"


def cache_key(
    source_id: str,
    cfg: PipelineConfig,
    mode: str = DEFAULT_MODE,
    audience: str = NEUTRAL_AUDIENCE,
) -> str:
    """Stable 16-hex key for (source, pipeline config, mode, audience level).

    `audience` defaults to NEUTRAL_AUDIENCE ("adult") and NOT to the product
    default ("teenager"), for the same reason `mode` defaults to "quality": this is
    the value that appends nothing, and every key written before the guardrail
    existed was written by an adult-prompt run. A caller that wants the
    submission's level must pass it — `resolve_audience` is the one place it
    comes from, and it must be resolved once and reused for both the key and the
    run, or the popup polls one key while the pipeline writes another.

    `|audience=<level>` rather than a bare `|<level>`, so a future reader cannot
    confuse the field positionally with `mode`.
    """
    cfg_hash = config_hash(cfg)
    payload = f"{source_id}|{cfg_hash}"
    if mode and mode != DEFAULT_MODE:
        payload += f"|{mode}"
    if audience and audience != NEUTRAL_AUDIENCE:
        payload += f"|audience={audience}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return digest[:16]


def lookup(cache_root: Path, key: str) -> CacheEntry | None:
    cache_dir = cache_root / key
    meta_path = cache_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    if not (cache_dir / "dub.wav").is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return CacheEntry(
        cache_key=key,
        cache_dir=cache_dir,
        source_id=meta.get("source_id", ""),
        source_url=meta.get("source_url"),
        duration=meta.get("duration"),
        created_at=float(meta.get("created_at", 0.0)),
        job_id=meta.get("job_id", ""),
    )


def lookup_by_url(
    cache_root: Path,
    url: str,
    cfg: PipelineConfig,
    mode: str = DEFAULT_MODE,
    audience: str = NEUTRAL_AUDIENCE,
) -> CacheEntry | None:
    """Resolve a URL to a cache entry under the current pipeline config + mode.

    Two layers: the `by-url.json` index lets us list keys ever produced for a
    given canonical URL; we then check whether any of those keys' config_hash
    matches the *current* cfg. This means a model swap that changes
    `config_hash` correctly misses the cache even if older entries exist.
    """
    canonical = canonicalize_url(url)
    src_id = f"url:{canonical}"
    expected_key = cache_key(src_id, cfg, mode, audience)
    return lookup(cache_root, expected_key)


def _load_by_url(cache_root: Path) -> dict[str, list[str]]:
    path = cache_root / _BY_URL_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_by_url(cache_root: Path, index: dict[str, list[str]]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / _BY_URL_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    tmp.replace(path)


def register_url(cache_root: Path, canonical_url: str, key: str) -> None:
    index = _load_by_url(cache_root)
    keys = index.setdefault(canonical_url, [])
    if key not in keys:
        keys.append(key)
    _save_by_url(cache_root, index)


def materialize(
    cache_root: Path,
    key: str,
    *,
    job_dir: Path,
    source_id: str,
    source_url: str | None,
    job_id: str,
    duration: float | None,
    pipeline_summary: dict[str, Any],
) -> CacheEntry:
    """Copy curated artifacts from a finished `runs/<job>/` dir into the cache.

    Only files load-bearing for playback + future-feature surface land in
    cache; the rest of `runs/<job>/` stays ephemeral and may be cleaned up
    independently.
    """
    cache_dir = cache_root / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_dub = job_dir / "dub.wav"
    if not src_dub.is_file():
        raise FileNotFoundError(f"No dub.wav in {job_dir} — pipeline must have failed before mux.")
    shutil.copy2(src_dub, cache_dir / "dub.wav")

    translated_json = job_dir / "translated.json"
    if translated_json.is_file():
        shutil.copy2(translated_json, cache_dir / "transcript.json")

    metadata = {
        "cache_key": key,
        "cache_version": CACHE_VERSION,
        "source_id": source_id,
        "source_url": source_url,
        "duration": duration,
        "job_id": job_id,
        "created_at": time.time(),
        "pipeline": pipeline_summary,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    if source_url:
        register_url(cache_root, canonicalize_url(source_url), key)

    return CacheEntry(
        cache_key=key,
        cache_dir=cache_dir,
        source_id=source_id,
        source_url=source_url,
        duration=duration,
        created_at=metadata["created_at"],
        job_id=job_id,
    )


def invalidate(cache_root: Path, key: str) -> bool:
    """Remove a cache entry and drop it from the by-url index.

    Used by the "Regenerate dub" path so the next submission goes through
    the pipeline instead of returning the stale cached `dub.wav`. Returns
    True if anything was removed.
    """
    removed = False
    cache_dir = cache_root / key
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)
        removed = True
    index = _load_by_url(cache_root)
    changed = False
    for canonical, keys in list(index.items()):
        if isinstance(keys, list) and key in keys:
            keys = [k for k in keys if k != key]
            if keys:
                index[canonical] = keys
            else:
                del index[canonical]
            changed = True
    if changed:
        _save_by_url(cache_root, index)
        removed = True
    return removed


def pipeline_summary(cfg: PipelineConfig) -> dict[str, Any]:
    """Compact representation of which candidates ran, for metadata.json."""
    return {
        name: {"id": cfg.entries.get(name, {}).get("id", "")}
        for name in STAGE_ORDER
    }


# ── watch-along (D3) per-section stream cache ──────────────────────────────
#
# A watch-along job streams dub in VAD-snapped sections instead of one final
# `dub.wav`. Each section's slice-relative WAV is written into the same
# `cache_root/<key>/` dir as `section_<NNN>.wav`, indexed by `sections.json`.
# Writing as we stream means a rewatch finds every section already on disk and
# plays instantly — the cache-synergy point of #10. The two layers (quality
# `dub.wav` vs watch-along `section_*.wav`) live under different keys (mode in
# the key) so they never collide.

STREAM_MANIFEST = "sections.json"


def stream_cache_dir(cache_root: Path, key: str) -> Path:
    d = cache_root / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stream_manifest_path(cache_root: Path, key: str) -> Path:
    return cache_root / key / STREAM_MANIFEST


def read_stream_manifest(cache_root: Path, key: str) -> dict[str, Any] | None:
    path = _stream_manifest_path(cache_root, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def init_stream_manifest(
    cache_root: Path,
    key: str,
    *,
    source_id: str,
    source_url: str | None,
    duration: float | None,
    pipeline_summary: dict[str, Any],
) -> dict[str, Any]:
    """Create (or reset) the section manifest for a watch-along job."""
    cache_dir = stream_cache_dir(cache_root, key)
    manifest = {
        "cache_key": key,
        "cache_version": CACHE_VERSION,
        "mode": "watch-along",
        "source_id": source_id,
        "source_url": source_url,
        "duration": duration,
        "complete": False,
        "created_at": time.time(),
        "pipeline": pipeline_summary,
        "sections": [],
    }
    _write_stream_manifest(cache_root, key, manifest)
    if source_url:
        register_url(cache_root, canonicalize_url(source_url), key)
    return manifest


def _write_stream_manifest(cache_root: Path, key: str, manifest: dict[str, Any]) -> None:
    path = _stream_manifest_path(cache_root, key)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.replace(path)


def append_section(
    cache_root: Path,
    key: str,
    *,
    index: int,
    start: float,
    end: float,
    src_wav: Path,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a finished section's WAV into the cache and add it to the manifest.

    Sections are keyed by their global `start` so an out-of-order section (e.g.
    one produced after a seek) replaces any prior section at the same start
    rather than duplicating it. Returns the updated manifest.

    `timing` (from `streaming/timing.py:section_timing`) is merged into the
    entry so a live stream records what it actually cost. It is additive and
    optional: an entry written without it, or by an older build, stays valid —
    readers must treat every timing key as absent-able. `index`, `start` and
    `end` are always taken from the arguments, never from `timing`, so the two
    cannot disagree.
    """
    cache_dir = stream_cache_dir(cache_root, key)
    fname = f"section_{index:03d}.wav"
    shutil.copy2(src_wav, cache_dir / fname)
    manifest = read_stream_manifest(cache_root, key) or {
        "cache_key": key, "complete": False, "sections": [],
    }
    sections = [s for s in manifest.get("sections", []) if abs(s.get("start", -1) - start) > 0.5]
    # The file name is derived from `index`, so two entries sharing an index
    # necessarily share a FILE — and the copy above has just overwritten it.
    # Any surviving entry with this index therefore now points at audio for a
    # different window and must go. This is what silently corrupted cache key
    # d5af5653d745ed66, where a restarted stream reset its counter to 0: indices
    # 1 and 2 each appeared twice with different windows, so the wrong audio
    # played at 154.9 s and 190.3 s. Dropping the stale entry here also repairs
    # an already-damaged manifest on the next append.
    stale = [s for s in sections if int(s.get("index", -1)) == index]
    if stale:
        log.warning(
            "stream cache %s: dropping %d stale section(s) reusing index %d "
            "(windows %s) — their file was just overwritten.",
            key, len(stale), index, [round(s.get("start", -1), 1) for s in stale],
        )
        sections = [s for s in sections if int(s.get("index", -1)) != index]
    entry: dict[str, Any] = dict(timing or {})
    entry.update({"index": index, "start": start, "end": end, "file": fname})
    sections.append(entry)
    sections.sort(key=lambda s: s["start"])
    manifest["sections"] = sections
    _write_stream_manifest(cache_root, key, manifest)
    return manifest


def finalize_stream(cache_root: Path, key: str) -> None:
    """Mark a watch-along stream complete (whole video dubbed → free rewatch)."""
    manifest = read_stream_manifest(cache_root, key)
    if manifest is None:
        return
    manifest["complete"] = True
    manifest["finalized_at"] = time.time()
    _write_stream_manifest(cache_root, key, manifest)
