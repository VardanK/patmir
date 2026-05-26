# Active output config

```toml
[pipeline]
output = "mkv_default"
```

Behavior:

- Container: MKV (cleanly handles multiple audio tracks + subs).
- Armenian audio is **track 0 (default)**, AAC-encoded.
- Original audio is **track 1**, copied (no re-encode).
- Armenian SRT is the default subtitle track.
- Metadata: `language=hye` + `title="Armenian (Patmir)"` on track 0; `language=<src>` + `title="Original"` on track 1.
