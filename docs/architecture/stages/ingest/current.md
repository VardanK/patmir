# Active ingest config

```toml
[pipeline]
ingest = "ffmpeg"
```

Configured in `patmir.toml [pipeline]`. No per-candidate options today — ffmpeg defaults to `-vn -ac 1 -ar 16000 -c:a pcm_s16le`.

## Requires

- `ffmpeg` and `ffprobe` on `PATH` (`brew install ffmpeg` on macOS).
