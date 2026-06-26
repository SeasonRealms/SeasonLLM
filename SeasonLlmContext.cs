// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace SeasonLLM;

public sealed class SeasonLlmContext : IDisposable
{
    private readonly SeasonLlmModel _model;
    private readonly int _batchSize;
    private readonly int _maxSequences;

    private IntPtr _handle;
    private bool _disposed;
    private int _position;

    internal SeasonLlmContext(SeasonLlmModel model, SeasonLlmContextOptions options)
    {
        SeasonLLM.EnsureSupported();
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);

        _model = model;
        _model.RentContext();

        try
        {
            var native = NativeMethods.llama_context_default_params();
            native.n_ctx = options.ContextSize;
            native.n_batch = options.BatchSize;
            native.n_ubatch = options.UBatchSize == 0 ? options.BatchSize : options.UBatchSize;
            native.n_seq_max = options.MaxSequences == 0 ? 1 : options.MaxSequences;
            native.n_threads = options.ThreadCount > 0 ? options.ThreadCount : Environment.ProcessorCount;
            native.n_threads_batch = options.BatchThreadCount > 0 ? options.BatchThreadCount : Environment.ProcessorCount;
            native.flash_attn_type = options.FlashAttention ? 1 : 0;
            native.offload_kqv = NativeMethods.ToNativeBool(options.OffloadKqv);
            native.no_perf = NativeMethods.ToNativeBool(options.NoPerf);

            _handle = NativeMethods.llama_init_from_model(model.Handle, native);
            if (_handle == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to create a llama context.");
            }

            NativeMethods.llama_set_n_threads(_handle, native.n_threads, native.n_threads_batch);
            _batchSize = Math.Max((int)NativeMethods.llama_n_batch(_handle), 1);
            _maxSequences = Math.Max((int)NativeMethods.llama_n_seq_max(_handle), 1);
        }
        catch
        {
            _model.ReturnContext();
            throw;
        }
    }

    public SeasonLlmModel Model => _model;

    public int ContextSize
    {
        get
        {
            ThrowIfDisposed();
            return (int)NativeMethods.llama_n_ctx(_handle);
        }
    }

    public int BatchSize
    {
        get
        {
            ThrowIfDisposed();
            return _batchSize;
        }
    }

    public void Reset()
    {
        ThrowIfDisposed();
        NativeMethods.llama_memory_clear(NativeMethods.llama_get_memory(_handle), true);
        _position = 0;
    }

    public SeasonLlmGenerationResult Complete(
        string prompt,
        SeasonLlmGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(prompt);
        return GenerateInternal(prompt, options ?? new SeasonLlmGenerationOptions(), false, null, cancellationToken);
    }

    public SeasonLlmGenerationResult Chat(
        IReadOnlyList<SeasonLlmChatMessage> messages,
        SeasonLlmGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(messages);

        var generationOptions = options ?? new SeasonLlmGenerationOptions();
        var usedChatTemplate = generationOptions.UseChatTemplate;
        var prompt = usedChatTemplate
            ? _model.ApplyChatTemplate(messages, generationOptions.AddAssistantGenerationPrompt, generationOptions.ChatTemplate)
            : SeasonLlmModel.BuildFallbackChatPrompt(messages, generationOptions.AddAssistantGenerationPrompt);

        return GenerateInternal(prompt, generationOptions, usedChatTemplate, null, cancellationToken);
    }

    public SeasonLlmGenerationResult CompleteStreaming(
        string prompt,
        Action<SeasonLlmGenerationChunk> onChunk,
        SeasonLlmGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(onChunk);
        return GenerateInternal(prompt, options ?? new SeasonLlmGenerationOptions(), false, onChunk, cancellationToken);
    }

    public SeasonLlmGenerationResult ChatStreaming(
        IReadOnlyList<SeasonLlmChatMessage> messages,
        Action<SeasonLlmGenerationChunk> onChunk,
        SeasonLlmGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(onChunk);

        var generationOptions = options ?? new SeasonLlmGenerationOptions();
        var usedChatTemplate = generationOptions.UseChatTemplate;
        var prompt = usedChatTemplate
            ? _model.ApplyChatTemplate(messages, generationOptions.AddAssistantGenerationPrompt, generationOptions.ChatTemplate)
            : SeasonLlmModel.BuildFallbackChatPrompt(messages, generationOptions.AddAssistantGenerationPrompt);

        return GenerateInternal(prompt, generationOptions, usedChatTemplate, onChunk, cancellationToken);
    }

    private SeasonLlmGenerationResult GenerateInternal(
        string prompt,
        SeasonLlmGenerationOptions options,
        bool usedChatTemplate,
        Action<SeasonLlmGenerationChunk>? onChunk,
        CancellationToken cancellationToken)
    {
        if (!options.UseExistingContext)
        {
            Reset();
        }

        var promptTokens = _model.Tokenize(prompt, options.AddSpecialTokens, options.ParseSpecialTokens);
        if (promptTokens.Length == 0)
        {
            throw new InvalidOperationException("The prompt produced no input tokens.");
        }

        using var abortBridge = CancellationBridge.Create(cancellationToken);
        NativeMethods.llama_set_abort_callback(
            _handle,
            abortBridge is null ? null : CancellationBridge.AbortThunk,
            abortBridge?.UserData ?? IntPtr.Zero);

        try
        {
            EvaluatePrompt(promptTokens, cancellationToken);

            if (options.MaxTokens <= 0)
            {
                return new SeasonLlmGenerationResult(
                    prompt,
                    string.Empty,
                    promptTokens,
                    [],
                    SeasonLlmFinishReason.MaxTokens,
                    null,
                    usedChatTemplate);
            }

            using var sampler = new SamplerHandle(CreateSampler(options));

            var generatedTokens = new List<int>(Math.Min(options.MaxTokens, 256));
            var finishReason = SeasonLlmFinishReason.MaxTokens;
            string? stopSequence = null;
            var generatedText = string.Empty;
            var lastStreamedText = string.Empty;

            for (var i = 0; i < options.MaxTokens; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var token = NativeMethods.llama_sampler_sample(sampler.Handle, _handle, -1);
                if (token == NativeMethods.LlamaTokenNull || NativeMethods.llama_vocab_is_eog(_model.VocabHandle, token))
                {
                    finishReason = SeasonLlmFinishReason.EndOfGeneration;
                    break;
                }

                generatedTokens.Add(token);

                var detokenizedText = _model.Detokenize(generatedTokens, removeSpecial: false, unparseSpecial: false);
                if (TryTrimStopSequence(detokenizedText, options.StopSequences, out stopSequence, out var visibleText))
                {
                    generatedText = visibleText;
                    EmitChunk(onChunk, token, visibleText, ref lastStreamedText);
                    finishReason = SeasonLlmFinishReason.StopSequence;
                    break;
                }

                generatedText = detokenizedText;
                EmitChunk(onChunk, token, generatedText, ref lastStreamedText);

                NativeMethods.llama_sampler_accept(sampler.Handle, token);
                EvaluateSingleToken(token, cancellationToken);
            }

            return new SeasonLlmGenerationResult(
                prompt,
                generatedText,
                promptTokens,
                generatedTokens,
                finishReason,
                stopSequence,
                usedChatTemplate);
        }
        finally
        {
            NativeMethods.llama_set_abort_callback(_handle, null, IntPtr.Zero);
        }
    }

    private void EvaluatePrompt(IReadOnlyList<int> promptTokens, CancellationToken cancellationToken)
    {
        using var batch = new LlamaBatchBuffer(_batchSize, _maxSequences);
        var tokenArray = promptTokens as int[] ?? promptTokens.ToArray();

        var offset = 0;
        while (offset < tokenArray.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var count = Math.Min(_batchSize, tokenArray.Length - offset);
            var requestLogits = offset + count >= tokenArray.Length;
            batch.SetTokens(tokenArray.AsSpan(offset, count), _position, requestLogits, sequenceId: 0);
            Decode(batch.Batch, cancellationToken);
            _position += count;
            offset += count;
        }
    }

    private void EvaluateSingleToken(int token, CancellationToken cancellationToken)
    {
        using var batch = new LlamaBatchBuffer(1, _maxSequences);
        Span<int> single = stackalloc int[1] { token };
        batch.SetTokens(single, _position, true, sequenceId: 0);
        Decode(batch.Batch, cancellationToken);
        _position++;
    }

    private void Decode(NativeMethods.NativeLlamaBatch batch, CancellationToken cancellationToken)
    {
        var result = NativeMethods.llama_decode(_handle, batch);
        if (result == 0)
        {
            return;
        }

        if (result == 2 || cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }

        if (result == 1)
        {
            throw new InvalidOperationException(
                "llama_decode could not find a KV slot for the batch. Reduce the batch size or increase the context size.");
        }

        if (result == -1)
        {
            throw new InvalidOperationException("llama_decode rejected the input batch.");
        }

        throw new InvalidOperationException($"llama_decode failed with native status {result}.");
    }

    private IntPtr CreateSampler(SeasonLlmGenerationOptions options)
    {
        var samplerParams = NativeMethods.llama_sampler_chain_default_params();
        var chain = NativeMethods.llama_sampler_chain_init(samplerParams);
        if (chain == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to create llama sampler chain.");
        }

        try
        {
            if (options.RepeatPenalty != 1.0f || options.FrequencyPenalty != 0f || options.PresencePenalty != 0f)
            {
                NativeMethods.llama_sampler_chain_add(
                    chain,
                    NativeMethods.llama_sampler_init_penalties(
                        options.RepeatLastTokens,
                        options.RepeatPenalty,
                        options.FrequencyPenalty,
                        options.PresencePenalty));
            }

            if (options.DoSample)
            {
                if (options.TopK > 0)
                {
                    NativeMethods.llama_sampler_chain_add(chain, NativeMethods.llama_sampler_init_top_k(options.TopK));
                }

                if (options.TopP is > 0f and < 1f)
                {
                    NativeMethods.llama_sampler_chain_add(chain, NativeMethods.llama_sampler_init_top_p(options.TopP, 1));
                }

                if (options.MinP > 0f)
                {
                    NativeMethods.llama_sampler_chain_add(chain, NativeMethods.llama_sampler_init_min_p(options.MinP, 1));
                }

                NativeMethods.llama_sampler_chain_add(chain, NativeMethods.llama_sampler_init_temp(options.Temperature));
                NativeMethods.llama_sampler_chain_add(chain, NativeMethods.llama_sampler_init_dist(options.Seed));
            }
            else
            {
                NativeMethods.llama_sampler_chain_add(chain, NativeMethods.llama_sampler_init_greedy());
            }

            return chain;
        }
        catch
        {
            NativeMethods.llama_sampler_free(chain);
            throw;
        }
    }

    private static void EmitChunk(
        Action<SeasonLlmGenerationChunk>? onChunk,
        int token,
        string accumulatedText,
        ref string lastStreamedText)
    {
        if (onChunk is null)
        {
            lastStreamedText = accumulatedText;
            return;
        }

        var chunkText = GetStreamingDelta(lastStreamedText, accumulatedText);
        if (chunkText.Length > 0)
        {
            onChunk(new SeasonLlmGenerationChunk(token, chunkText, accumulatedText));
        }

        lastStreamedText = accumulatedText;
    }

    private static string GetStreamingDelta(string previousText, string currentText)
    {
        if (string.IsNullOrEmpty(previousText))
        {
            return currentText;
        }

        if (currentText.StartsWith(previousText, StringComparison.Ordinal))
        {
            return currentText[previousText.Length..];
        }

        var commonLength = 0;
        var max = Math.Min(previousText.Length, currentText.Length);
        while (commonLength < max && previousText[commonLength] == currentText[commonLength])
        {
            commonLength++;
        }

        return currentText[commonLength..];
    }

    private static bool TryTrimStopSequence(
        string text,
        IReadOnlyList<string>? stopSequences,
        out string? matchedStopSequence,
        out string trimmedText)
    {
        matchedStopSequence = null;
        trimmedText = text;
        if (stopSequences is null || stopSequences.Count == 0 || text.Length == 0)
        {
            return false;
        }

        foreach (var stopSequence in stopSequences)
        {
            if (string.IsNullOrEmpty(stopSequence))
            {
                continue;
            }

            if (text.EndsWith(stopSequence, StringComparison.Ordinal))
            {
                trimmedText = text[..^stopSequence.Length];
                matchedStopSequence = stopSequence;
                return true;
            }
        }

        return false;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_handle != IntPtr.Zero)
        {
            NativeMethods.llama_free(_handle);
            _handle = IntPtr.Zero;
        }

        _model.ReturnContext();
        GC.SuppressFinalize(this);
    }

    ~SeasonLlmContext()
    {
        if (_handle != IntPtr.Zero)
        {
            NativeMethods.llama_free(_handle);
            _handle = IntPtr.Zero;
            _model.ReturnContext();
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    private sealed class SamplerHandle : IDisposable
    {
        public SamplerHandle(IntPtr handle)
        {
            Handle = handle;
        }

        public IntPtr Handle { get; private set; }

        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
            {
                NativeMethods.llama_sampler_free(Handle);
                Handle = IntPtr.Zero;
            }
        }
    }
}
