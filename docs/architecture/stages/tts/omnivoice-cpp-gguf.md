# OmniVoice GGUF via omnivoice.cpp (tts)

- **Repo / version:** `Serveurperso/OmniVoice-GGUF` (canonical; `fablevi/OmniVoice-GGUF-clone` is a byte-identical duplicate). Runtime: `omnivoice.cpp` (C++17/GGML port, https://github.com/ServeurpersoCom/omnivoice.cpp, MIT). Same upstream model as our MLX OmniVoice (k2-fsa/OmniVoice + Higgs Audio v2 tokenizer).
- **Variants evaluated:** base+tokenizer GGUF at **BF16, Q8_0, Q4_K_M** (also F32 available).
- **Quantization available:** F32 / BF16 / Q8_0 / Q4_K_M.
- **Apple Silicon:** builds with Metal (`cmake -DGGML_METAL=ON`); ggml submodule. No prebuilt binary (no GitHub releases / Homebrew) — must build from source (small, ~few-min build).
- **Steps:** MaskGIT `mg_num_step = 32` is a fixed in-source default (`src/omnivoice.cpp:189`); **no CLI flag** — so all runs are 32 steps.
- **Cloning:** `omnivoice-tts --model <base.gguf> --codec <tokenizer.gguf> --ref-wav <ref.wav> --ref-text <ref.txt> --lang hy --seed N -o out.wav < prompt.txt`. `--lang` resolves ISO codes (`lang-map.h`: "armenian"→"hy"). 24 kHz mono out.
- **License / gating:** Apache 2.0 (model), MIT (runtime). Ungated.
- **Status in pipeline:** **rejected.**
- **Last evaluated:** 2026-06-17.
- **Reason rejected:**
  - **Slower than our MLX bf16.** Benchmark (Apple Silicon, Metal, 32 steps, identical clean adult ref + Armenian prompt): omnivoice.cpp internal RTF (gen+decode) ≈ **1.89–2.04** across BF16/Q8_0/Q4_K_M (wall RTF 2.1–2.7 incl. load + per-process Metal kernel JIT). Our `omnivoice_mlx_bf16` synth RTF ≈ **1.1–1.5** on the same prompts — roughly **2× faster**. Quant level barely moves omnivoice.cpp speed (compute-bound on Metal), so Q4_K_M buys no speed over BF16.
  - **Quality not better.** Author ear-test (same ref + prompt, all four side by side): **MLX bf16 won the ear-test** over cpp BF16/Q8_0/Q4_K_M. The cpp variants also rendered the same text noticeably longer/slower-paced.
  - **Added dependency cost.** Requires a C++/Metal build + ggml submodule + ~3.2 GB of GGUFs, for no speed or quality gain.
- **Revisit:** if omnivoice.cpp exposes a tunable step count and a higher-step run is clearly better, if it ships a prebuilt Metal binary AND closes the speed gap, or if our MLX OmniVoice path breaks and we need a non-MLX local fallback. Spike artifacts (`spike/omnivoice.cpp`, `spike/omni_cpp_bench`) are throwaway.
