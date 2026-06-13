"""Probe: can we load the audeering wav2vec2 age-gender model and profile our
ref clips? Decides whether #7 profiling uses the model or a pitch fallback.

Model: audeering/wav2vec2-large-robust-24-ft-age-gender
  - age:    sigmoid regression, ×100 = years
  - gender: logits [female, male, child]

Run: .venv/bin/python spike/d3_throughput/probe_age_gender.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"
MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        # Newer transformers expects these on every PreTrainedModel; the
        # original audeering custom class predates them. No tied weights here.
        _tied_weights_keys = []
        all_tied_weights_keys = {}

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values).last_hidden_state
            hidden = torch.mean(hidden, dim=1)
            return self.age(hidden), self.gender(hidden)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} on {device} ...", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    model = AgeGenderModel.from_pretrained(MODEL).to(device).eval()
    print("loaded.", flush=True)

    refs = json.loads((RUN / "diarize/refs.json").read_text())
    GENDERS = ["female", "male", "child"]
    for key, v in refs.items():
        y, sr = sf.read(v["ref_audio"], dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y[: sr * 8]  # cap analysis length
        inp = processor(y, sampling_rate=sr, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            age_logit, gender_logit = model(inp)
        age_years = float(torch.sigmoid(age_logit)[0, 0]) * 100
        gprobs = torch.softmax(gender_logit, dim=1)[0]
        gender = GENDERS[int(torch.argmax(gprobs))]
        print(f"{key:24} age≈{age_years:5.1f}y  gender={gender:6} "
              f"(f={gprobs[0]:.2f} m={gprobs[1]:.2f} c={gprobs[2]:.2f})", flush=True)

    print("\nPROBE OK")


if __name__ == "__main__":
    main()
