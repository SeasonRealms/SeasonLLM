# SeasonLLM

`SeasonLLM` is a small .NET wrapper around `llama.cpp` for local GGUF text generation.

It follows the same minimal style as `SeasonImage`:

- Static entry point for process-wide initialization and logging
- Separate model and context handles
- Simple completion and chat APIs
- Streaming token callbacks
- Tokenize / detokenize helpers
- Windows native runtimes can be bundled in `runtimes/win-x64/native`

https://github.com/SeasonRealms/SeasonLLM

## Current Scope

First-stage managed wrapper focus:

- GGUF model loading
- Context creation
- Prompt completion
- Chat completion via `llama_chat_apply_template`
- Streaming output
- Cancellation via `CancellationToken`
- Tokenization helpers

## Quick Start

```csharp
using SeasonLLM;

SeasonLLM.SetLogCallback((level, text) =>
{
    Console.WriteLine($"[{level}] {text}");
});

using var model = SeasonLLM.CreateModel(new SeasonLlmModelOptions
{
    ModelPath = @"C:\Models\Qwen3-4B-Instruct-Q4_K_M.gguf",
    Backend = "cpu",
    GpuLayers = 0,
    UseMmap = true
});

using var ctx = model.CreateContext(new SeasonLlmContextOptions
{
    ContextSize = 8192,
    BatchSize = 512,
    ThreadCount = Environment.ProcessorCount,
    FlashAttention = false
});

var result = ctx.Chat(
[
    new SeasonLlmChatMessage("system", "You are a concise assistant."),
    new SeasonLlmChatMessage("user", "Explain what GGUF is in one paragraph.")
],
new SeasonLlmGenerationOptions
{
    MaxTokens = 256,
    Temperature = 0.7f,
    TopK = 40,
    TopP = 0.95f,
    StopSequences = ["<|im_end|>"]
});

Console.WriteLine(result.Text);
```

## Backend Selection

```csharp
var backends = SeasonLLM.GetAvailableBackends();
Console.WriteLine(string.Join(", ", backends));

using var model = SeasonLLM.CreateModel(new SeasonLlmModelOptions
{
    ModelPath = @"C:\Models\Qwen3-4B-Instruct-Q4_K_M.gguf",
    Backend = "vulkan0",
    // Also accepts values such as "cpu", "cuda0", "vulkan0,cpu"
    GpuLayers = -1,
    UseMmap = true
});
```

`Backend` limits which ggml devices `llama.cpp` can use instead of letting it auto-pick all available devices.
`ParamsBackend` is accepted for compatibility with `SeasonImage`-style configuration and is merged into the same native device list.

## Streaming

```csharp
ctx.CompleteStreaming(
    "Write a short poem about local AI.",
    chunk => Console.Write(chunk.Text),
    new SeasonLlmGenerationOptions
    {
        MaxTokens = 128,
        Temperature = 0.8f,
        TopP = 0.95f
    });
```

## Notes

- `llama.cpp` logging is global, just like `SeasonImage` progress and log callbacks.
- The current wrapper keeps the public API intentionally small.
- Advanced features such as embeddings, LoRA, state save/load, grammar-constrained output, and KV sequence management can be added later on top of the same native binding layer.
