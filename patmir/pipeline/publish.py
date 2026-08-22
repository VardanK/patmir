"""Publish a finished job's deliverables to `[storage].library_root`.

`runs_root/<job_id>/` is the working directory: dozens of intermediate WAVs,
per-stage JSON sidecars, and the finished container. `library_root/<job_id>/`
is the curated export — just the files worth keeping and watching:

    library_root/<job_id>/out.mkv       final container (Armenian default audio)
    library_root/<job_id>/out.hy.srt    Armenian subtitles
    library_root/<job_id>/dub.wav       Armenian audio on its own

Runs in the runner after the `output` stage rather than inside the output
candidate, so `stages/output/mkv.py` stays destination-agnostic (it always
writes to `ctx.job_dir`). Same pattern as the other post-stage runner steps
(`pick_refs`, `apply_speaker_tags`).

**Copy, not move.** The job dir has to stay intact: `--resume-from` reads
absolute paths back out of the sidecar JSONs, and the server's
`cache.materialize` pulls `dub.wav` from there after the run returns.

Each file is written to `<name>.partial` and then renamed into place. The rename
happens inside the destination directory, so it stays a same-filesystem
operation even when the library is a network share — without it, an interrupted
transfer would leave a truncated `out.mkv` that looks like a valid file.
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Files copied out of the job dir, in the order they're published. Missing
# entries are skipped (e.g. an audio-only run still has all three, but a job
# stopped before `output` has no MKV).
DELIVERABLES = ("out.mkv", "out.hy.srt", "dub.wav")

_PARTIAL_SUFFIX = ".partial"


def publish_job(job_dir: Path, job_id: str, library_root: Path) -> Path:
    """Copy `job_dir`'s deliverables into `library_root/<job_id>/`.

    Returns the destination directory. Raises if nothing could be published —
    an empty publish dir on the share would be worse than a loud failure.
    """
    dest_dir = library_root / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    published: list[Path] = []
    for name in DELIVERABLES:
        src = job_dir / name
        if not src.is_file():
            continue
        published.append(_copy_atomic(src, dest_dir / name))

    if not published:
        raise FileNotFoundError(
            f"Nothing to publish from {job_dir} — expected one of {', '.join(DELIVERABLES)}."
        )
    return dest_dir


def _copy_atomic(src: Path, dest: Path) -> Path:
    """Copy `src` to `dest` via a sibling `.partial` file, then rename."""
    partial = dest.with_name(dest.name + _PARTIAL_SUFFIX)
    try:
        # copyfile, not copy2: copystat can raise or silently no-op on SMB/NFS,
        # and the mtime isn't load-bearing for anything downstream.
        shutil.copyfile(src, partial)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return dest
