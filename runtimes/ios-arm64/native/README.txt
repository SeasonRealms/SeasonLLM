Platform: ios-arm64 (multi-backend xcframework carrying the device and arm64-simulator slices)
Configuration: Release
LLAMA_REPO: https://github.com/SeasonRealms/llama.cpp
LLAMA_REF: master
GGML_REPO: https://github.com/SeasonRealms/ggml
GGML_REF: master
Target triples: arm64-apple-ios17.0 (device), arm64-apple-ios17.0-simulator (simulator)
Linkage: static (BUILD_SHARED_LIBS=OFF, GGML_BACKEND_DL=OFF), merged into a single libllama.a per slice
Enabled backends: cpu (armv8.2-a+dotprod), metal, blas=true
GGML_MAX_NAME: 128

This build packages a single self-contained llama.cpp runtime for iOS.
Two slices only, both arm64: ios-arm64 (device) and ios-arm64-simulator (Apple Silicon Macs). One arm64 iOS target does NOT also cover the simulator - device and simulator are distinct Mach-O platforms; no x86_64 simulator slice is produced, Intel Macs are out of scope.
On iOS both the llama and mtmd P/Invoke bindings are meant to bind as __Internal (unlike the libllama/libmtmd dylibs used on Mac Catalyst), so the app statically links libllama.a via NativeReference (Kind=Static); the merged archive exports the llama, mtmd and ggml entry points from one file.
Static linkage: llama, mtmd, ggml and every backend are archived into the single libllama.a and registered at runtime by ggml's backend registry; nothing is dlopen'd.
Self-contained: a complete ggml build is merged in. If several self-contained ggml-based archives (ggml, llama, qwentts, ...) are linked into the same app, keep exactly one ForceLoad=True owner; otherwise the duplicate ggml objects clash at link time.
This build compiles ggml with GGML_MAX_NAME=128 (the upstream default is 64) so the ggml_tensor layout matches the other ggml-based runtimes linked into the same app (qwentts, stable-diffusion and SeasonGGML, which all use 128).
CPU targets the M1 baseline (armv8.2-a+dotprod), supported by every A12+ device and every Apple Silicon simulator.
Metal uses GGML_METAL_EMBED_LIBRARY so the shader source travels inside libllama.a; no default.metallib has to be deployed or located in the app bundle.
BLAS uses Apple's Accelerate framework (GGML_BLAS_VENDOR=Apple), which is available on both iOS devices and simulators.
CUDA and Vulkan are unavailable on iOS and are explicitly disabled.
Command line tools (llama-cli, llama-server) are not built for iOS: this artifact is the native library consumed by SeasonLLM through P/Invoke.

libllama.a sha256 (ios-arm64): d6cadeaaf34be8de2dacfe17815525e9954c8427e5b350153f0cc842dfd4cb6b
libllama.a sha256 (ios-arm64-simulator): 968676e25f13f08431ebe9ba8787f1a8813c9d7048dd043d7880adb31024fc13
Artifact contents:
.
./ggml.txt
./libllama.xcframework
./libllama.xcframework/Info.plist
./libllama.xcframework/ios-arm64
./libllama.xcframework/ios-arm64-simulator
./libllama.xcframework/ios-arm64-simulator/libllama.a
./libllama.xcframework/ios-arm64/libllama.a
./llama-cpp.h
./llama.cpp.txt
./llama.h
./README.txt
