// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

public sealed class SeasonLlmChatMessage
{
    public SeasonLlmChatMessage(string role, string content)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(role);
        ArgumentNullException.ThrowIfNull(content);

        Role = role;
        Content = content;
    }

    public string Role { get; }
    public string Content { get; }
}

public sealed class SeasonLlmGenerationChunk
{
    public SeasonLlmGenerationChunk(int tokenId, string text, string accumulatedText)
    {
        TokenId = tokenId;
        Text = text;
        AccumulatedText = accumulatedText;
    }

    public int TokenId { get; }
    public string Text { get; }
    public string AccumulatedText { get; }
}

public sealed class SeasonLlmGenerationResult
{
    public SeasonLlmGenerationResult(
        string prompt,
        string text,
        IReadOnlyList<int> promptTokens,
        IReadOnlyList<int> generatedTokens,
        SeasonLlmFinishReason finishReason,
        string? stopSequence,
        bool usedChatTemplate)
    {
        Prompt = prompt;
        Text = text;
        PromptTokens = promptTokens;
        GeneratedTokens = generatedTokens;
        FinishReason = finishReason;
        StopSequence = stopSequence;
        UsedChatTemplate = usedChatTemplate;
    }

    public string Prompt { get; }
    public string Text { get; }
    public IReadOnlyList<int> PromptTokens { get; }
    public IReadOnlyList<int> GeneratedTokens { get; }
    public SeasonLlmFinishReason FinishReason { get; }
    public string? StopSequence { get; }
    public bool UsedChatTemplate { get; }
    public int PromptTokenCount => PromptTokens.Count;
    public int GeneratedTokenCount => GeneratedTokens.Count;
}
