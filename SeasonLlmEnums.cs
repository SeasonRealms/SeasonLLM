// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

public enum SeasonLlmLogLevel
{
    None = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Continue = 5
}

public enum SeasonLlmFinishReason
{
    EndOfGeneration,
    StopSequence,
    MaxTokens
}
