# Active emotion config

```toml
[pipeline]
emotion = "emotion2vec_plus_base"

[[stages.emotion.candidates]]
id        = "emotion2vec_plus_base"
module    = "patmir.pipeline.stages.emotion.emotion2vec"
model_id  = "iic/emotion2vec_plus_base"
device    = "auto"
min_confidence = 0.5   # below this → emotion bucket forced to "neutral"

[emotion_bins]
# Maps the 9 emotion2vec classes → 5 ref-clip bins. Each key is a SER class
# label; each value is the bin name. Bins are: neutral, happy, sad, angry, fearful.
angry      = "angry"
disgusted  = "angry"
fearful    = "fearful"
happy      = "happy"
neutral    = "neutral"
other      = "neutral"
sad        = "sad"
surprised  = "happy"
unknown    = "neutral"

[emotion_guidance]
# OmniVoice guidance_scale per emotion bin. Higher = more faithful to ref + text.
neutral = 0.7
happy   = 0.9
sad     = 0.6
angry   = 0.9
fearful = 0.8
```

## Bin design

5 bins is a tradeoff between expressive coverage and data sparsity per character. With more bins, each `(speaker, emotion)` cell gets fewer candidate turns to draw a ref clip from, raising the chance of an empty bin. 5 bins keeps every "expected" character archetype (happy/sad/angry/fearful) plus a neutral fallback.
