# Third-Party Sources

- `third_party/sglang-omni` includes SGLang-Omni under its Apache-2.0 license. Its runtime includes model implementations and vendored components with their own notices. SolarX licensing information remains under `sglang_omni/vendor/solarx`.
- The browser demo and low-delay AVC encoder adapt Apache-2.0 code from the Duo vLLM-Omni inference branch. Existing contributor notices remain in the source. The SGLang-Omni runtime does not require vLLM at runtime.
- The optional FP8 FA3 build uses a patch under `benchmarks/duo/fp8/csrc` and separately obtained FlashAttention and CUTLASS sources. Their respective licenses apply. No prebuilt shared library is included.
- `third_party/duo` contains Duo model compatibility sources with their original source headers and file hashes. The source revision did not include a root license file. See its `NOTICE.md`; inclusion here does not relicense that directory or grant additional redistribution rights.
- Model weights, tokenizers, calibration profiles, and example media are obtained separately and remain subject to their providers' terms.
