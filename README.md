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
- Grammar-constrained output
- Single LoRA adapter loading at model initialization
- Single-image question answering via `mmproj` (Gemma 4 E4B, Qwen3.5, and other GGUF vision pairs)

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

## Grammar / JSON

```csharp
var grammarResult = ctx.Chat(
[
    new SeasonLlmChatMessage("user", "Return a JSON object with title and priority.")
],
new SeasonLlmGenerationOptions
{
    MaxTokens = 128,
    Temperature = 0.2f,
    JsonSchema = """
    {
      "type": "object",
      "properties": {
        "title": { "type": "string", "minLength": 1, "maxLength": 80 },
        "priority": { "type": "string", "enum": ["low", "medium", "high"] }
      },
      "required": ["title", "priority"],
      "additionalProperties": false
    }
    """
});

Console.WriteLine(grammarResult.Text);
```

You can also pass raw GBNF directly:

```csharp
var result = ctx.Complete(
    "Respond with yes or no only.",
    new SeasonLlmGenerationOptions
    {
        MaxTokens = 8,
        Grammar = """
        root ::= "yes" | "no"
        """
    });
```

`JsonOutput = true` enables unconstrained JSON object output without a schema.

The current `JsonSchema` converter supports a practical subset:

- `type`
- `properties`
- `required`
- `additionalProperties: false`
- `items`
- `minItems` / `maxItems`
- `minLength` / `maxLength`
- `enum`
- `const`

Unsupported schema keywords currently throw `NotSupportedException`.

## LoRA

```csharp
using var model = SeasonLLM.CreateModel(new SeasonLlmModelOptions
{
    ModelPath = @"C:\Models\Qwen3-4B-Instruct-Q4_K_M.gguf",
    LoraPath = @"C:\Models\qwen3-writing-style-lora.gguf",
    LoraScale = 1.0f,
    Backend = "cuda0",
    GpuLayers = -1
});

using var ctx = model.CreateContext(new SeasonLlmContextOptions
{
    ContextSize = 8192,
    BatchSize = 512
});

var result = ctx.Complete("Write a short product tagline.");
Console.WriteLine(result.Text);
```

The current wrapper applies at most one LoRA adapter per model and automatically enables it for every context created from that model.

## Single Image

The wrapper exposes a single-image multimodal path for any text model loaded together with a matching `mmproj` projector file, such as `gemma-4-E4B-it` or `Qwen3.5-0.8B`.
Call `CompleteImage()` or `CompleteImageStreaming()` with encoded image bytes such as PNG or JPEG.
The prompt is built from the model's own chat template when available, with an explicit ChatML fallback, so different model families are formatted correctly.

```csharp
using var model = SeasonLLM.CreateModel(new SeasonLlmModelOptions
{
    ModelPath = @"C:\Models\Qwen3.5-0.8B-Q4_K_M.gguf",
    MmprojPath = @"C:\Models\mmproj-BF16.gguf",
    MmprojUseGpu = true,
    ImageMaxTokens = 560,
    Backend = "cuda0",
    GpuLayers = -1
});

using var ctx = model.CreateContext();

var result = ctx.CompleteImageStreaming(
    "Describe the UI shown in this screenshot.",
    File.ReadAllBytes(@"C:\Images\screen.png"),
    chunk => Console.Write(chunk.Text),
    systemPrompt: "You are a concise and helpful assistant.");
```

Current limitations:

- The `mmproj` file must match the text model; an incompatible pair fails at mtmd initialization.
- Only a single image is supported per request.
- The multimodal API accepts encoded image bytes, not decoded RGBA buffers.
- Existing text `Chat()` and `Complete()` behavior is unchanged.

## Notes

- `llama.cpp` logging is global, just like `SeasonImage` progress and log callbacks.
- The current wrapper keeps the public API intentionally small.
- Advanced features such as embeddings, richer multimodal input, state save/load, and KV sequence management can be added later on top of the same native binding layer.
