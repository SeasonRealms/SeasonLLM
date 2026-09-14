LLAMA_REPO: https://github.com/SeasonRealms/llama.cpp
LLAMA_REF: master
GGML_REPO: https://github.com/SeasonRealms/ggml
GGML_REF: master
Configuration: Release
RID: maccatalyst-arm64
Target triple: arm64-apple-ios15.0-macabi
Linkage: static (BUILD_SHARED_LIBS=OFF, GGML_BACKEND_DL=OFF), merged into a single libllama.dylib
Enabled backends: cpu (armv8.2-a+dotprod), metal, blas=true

This build packages a single self-contained llama.cpp runtime for maccatalyst-arm64.
Apple Silicon only: no x86_64 slice is produced, so this runtime does not load on Intel Macs. Add a maccatalyst-x64 build if Intel support is needed again.
Static linkage: llama, ggml and every backend are archived into libllama.dylib and registered at runtime by ggml's backend registry; nothing is dlopen'd, so the app bundle contains a single Mach-O to sign.
libmtmd.dylib is a thin re-export shim forwarding every symbol to libllama.dylib, so DllImport("mtmd") and DllImport("llama") share one copy of llama/ggml state.
CPU targets the M1 baseline (armv8.2-a+dotprod). GGML_CPU_ALL_VARIANTS is intentionally NOT used because it requires GGML_BACKEND_DL; M1 and every later Apple Silicon chip run this baseline.
Metal uses GGML_METAL_EMBED_LIBRARY so the shader source travels inside libllama.dylib; no default.metallib has to be deployed or located in the app bundle.
Metal targets the Apple Silicon unified-memory GPU only; Intel integrated and AMD discrete GPUs are out of scope for this artifact.
CUDA and Vulkan are unavailable on Mac Catalyst and are explicitly disabled.
Command line tools (llama-cli, llama-server) are not built for Mac Catalyst: this artifact is the native library consumed by SeasonLLM through P/Invoke.

libllama.dylib sha256: fbc3fa8c95d6a0c174e0543d1c97a0abe10d6211f343e9c7eec0ad53bf51d54f
Artifact contents:
ggml.txt
libllama.dylib
libmtmd.dylib
llama-cpp.h
llama.cpp.txt
llama.h
README.txt
