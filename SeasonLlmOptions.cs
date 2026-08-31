// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

public sealed class SeasonLlmModelOptions
{
    public string ModelPath { get; set; } = string.Empty;
    public string? LoraPath { get; set; }
    public float LoraScale { get; set; } = 1.0f;
    public string? MmprojPath { get; set; }
    public bool MmprojUseGpu { get; set; } = true;
    public int ImageMinTokens { get; set; }
    public int ImageMaxTokens { get; set; }
    public string? Backend { get; set; }
    public string? ParamsBackend { get; set; }
    public int GpuLayers { get; set; }
    public int MainGpu { get; set; }
    public bool UseMmap { get; set; } = true;
    public bool UseMlock { get; set; }
    public bool UseDirectIo { get; set; }
    public bool CheckTensors { get; set; }
    public bool VocabularyOnly { get; set; }
    public Action<float>? LoadProgressCallback { get; set; }
}

public sealed class SeasonLlmContextOptions
{
    public uint ContextSize { get; set; }
    public uint BatchSize { get; set; } = 512;
    public uint UBatchSize { get; set; } = 512;
    public uint MaxSequences { get; set; } = 1;
    public int ThreadCount { get; set; } = Environment.ProcessorCount;
    public int BatchThreadCount { get; set; } = Environment.ProcessorCount;
    public bool FlashAttention { get; set; }
    public bool OffloadKqv { get; set; } = true;
    public bool NoPerf { get; set; }
}

public sealed class SeasonLlmGenerationOptions
{
    public int MaxTokens { get; set; } = 256;
    public uint Seed { get; set; } = uint.MaxValue;
    public bool DoSample { get; set; } = true;
    public float Temperature { get; set; } = 0.8f;
    public int TopK { get; set; } = 40;
    public float TopP { get; set; } = 0.95f;
    public float MinP { get; set; } = 0.05f;
    public float RepeatPenalty { get; set; } = 1.1f;
    public int RepeatLastTokens { get; set; } = 64;
    public float FrequencyPenalty { get; set; }
    public float PresencePenalty { get; set; }
    public IReadOnlyList<string>? StopSequences { get; set; }
    public bool AddSpecialTokens { get; set; } = true;
    public bool ParseSpecialTokens { get; set; } = true;
    public bool UseExistingContext { get; set; }
    public bool UseChatTemplate { get; set; } = true;
    public bool AddAssistantGenerationPrompt { get; set; } = true;
    public string? ChatTemplate { get; set; }
    public string? Grammar { get; set; }
    public string GrammarRoot { get; set; } = "root";
    public bool JsonOutput { get; set; }
    public string? JsonSchema { get; set; }
}
