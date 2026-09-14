// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

public sealed class SeasonLlmContext : IDisposable
{
    private readonly SeasonLlmModel _model;
    private readonly int _batchSize;
    private readonly int _maxSequences;
    private readonly int _threadCount;
    private readonly int _flashAttentionType;

    private IntPtr _handle;
    private MtmdContextHandle? _mtmdContext;
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

            _model.ApplyConfiguredLora(_handle);
            NativeMethods.llama_set_n_threads(_handle, native.n_threads, native.n_threads_batch);
            _batchSize = Math.Max((int)NativeMethods.llama_n_batch(_handle), 1);
            _maxSequences = Math.Max((int)NativeMethods.llama_n_seq_max(_handle), 1);
            _threadCount = native.n_threads;
            _flashAttentionType = native.flash_attn_type;
        }
        catch
        {
            if (_handle != IntPtr.Zero)
            {
                NativeMethods.llama_free(_handle);
                _handle = IntPtr.Zero;
            }

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

    public SeasonLlmGenerationResult CompleteImage(
        string prompt,
        byte[] imageBytes,
        SeasonLlmGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        ArgumentNullException.ThrowIfNull(imageBytes);
        if (imageBytes.Length == 0)
        {
            throw new ArgumentException("Image bytes cannot be empty.", nameof(imageBytes));
        }

        return GenerateImageInternal(prompt, imageBytes, options ?? new SeasonLlmGenerationOptions(), null, cancellationToken);
    }

    public SeasonLlmGenerationResult CompleteImageStreaming(
        string prompt,
        byte[] imageBytes,
        Action<SeasonLlmGenerationChunk> onChunk,
        SeasonLlmGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        ArgumentNullException.ThrowIfNull(imageBytes);
        ArgumentNullException.ThrowIfNull(onChunk);
        if (imageBytes.Length == 0)
        {
            throw new ArgumentException("Image bytes cannot be empty.", nameof(imageBytes));
        }

        return GenerateImageInternal(prompt, imageBytes, options ?? new SeasonLlmGenerationOptions(), onChunk, cancellationToken);
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
            return GenerateFromCurrentState(prompt, promptTokens, options, usedChatTemplate, onChunk, cancellationToken);
        }
        finally
        {
            NativeMethods.llama_set_abort_callback(_handle, null, IntPtr.Zero);
        }
    }

    private SeasonLlmGenerationResult GenerateImageInternal(
        string prompt,
        byte[] imageBytes,
        SeasonLlmGenerationOptions options,
        Action<SeasonLlmGenerationChunk>? onChunk,
        CancellationToken cancellationToken)
    {
        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 1: EnsureGemma4SingleImageSupport");
        EnsureGemma4SingleImageSupport();

        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 2: Reset (if needed)");
        if (!options.UseExistingContext)
        {
            Reset();
        }

        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 3: Create abortBridge");
        using var abortBridge = CancellationBridge.Create(cancellationToken);
        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 4: llama_set_abort_callback");
        NativeMethods.llama_set_abort_callback(
            _handle,
            abortBridge is null ? null : CancellationBridge.AbortThunk,
            abortBridge?.UserData ?? IntPtr.Zero);

        try
        {
            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 5: Create Utf8StringArena");
            using var strings = new Utf8StringArena();
            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 6: CreateImageBitmap");
            using var bitmap = CreateImageBitmap(imageBytes);
            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 7: CreateImagePromptChunks");
            using var chunks = CreateImagePromptChunks(prompt, bitmap.BitmapHandle, options, strings, out var promptText);

            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 8: mtmd_helper_get_n_tokens");
            var promptTokenCount = checked((int)MtmdNativeMethods.mtmd_helper_get_n_tokens(chunks.Handle));
            var promptTokens = promptTokenCount > 0 ? new int[promptTokenCount] : [];
            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 9: EvaluateImagePrompt");
            EvaluateImagePrompt(chunks.Handle, options.MaxTokens > 0, cancellationToken);

            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] step 10: GenerateFromCurrentState");
            return GenerateFromCurrentState(promptText, promptTokens, options, usedChatTemplate: true, onChunk, cancellationToken);
        }
        finally
        {
            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] finally: clear abort_callback");
            NativeMethods.llama_set_abort_callback(_handle, null, IntPtr.Zero);
        }
    }

    private SeasonLlmGenerationResult GenerateFromCurrentState(
        string prompt,
        IReadOnlyList<int> promptTokens,
        SeasonLlmGenerationOptions options,
        bool usedChatTemplate,
        Action<SeasonLlmGenerationChunk>? onChunk,
        CancellationToken cancellationToken)
    {
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
        var decoder = Encoding.UTF8.GetDecoder();
        var streamedText = new StringBuilder();
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

            var decodedChunk = DecodeUtf8Bytes(decoder, _model.TokenToPieceBytes(token), flush: false);
            if (decodedChunk.Length > 0)
            {
                streamedText.Append(decodedChunk);
                var accumulatedText = streamedText.ToString();
                if (TryTrimStopSequence(accumulatedText, options.StopSequences, out stopSequence, out var visibleText))
                {
                    generatedText = visibleText;
                    EmitChunk(onChunk, token, visibleText, ref lastStreamedText);
                    finishReason = SeasonLlmFinishReason.StopSequence;
                    break;
                }

                generatedText = accumulatedText;
                EmitChunk(onChunk, token, generatedText, ref lastStreamedText);
            }

            NativeMethods.llama_sampler_accept(sampler.Handle, token);
            EvaluateSingleToken(token, cancellationToken);
        }

        if (finishReason != SeasonLlmFinishReason.StopSequence)
        {
            var flushedText = DecodeUtf8Bytes(decoder, [], flush: true);
            if (flushedText.Length > 0)
            {
                streamedText.Append(flushedText);
                var accumulatedText = streamedText.ToString();
                if (TryTrimStopSequence(accumulatedText, options.StopSequences, out stopSequence, out var visibleText))
                {
                    generatedText = visibleText;
                    EmitChunk(onChunk, generatedTokens.Count > 0 ? generatedTokens[^1] : 0, visibleText, ref lastStreamedText);
                    finishReason = SeasonLlmFinishReason.StopSequence;
                }
                else
                {
                    generatedText = accumulatedText;
                    EmitChunk(onChunk, generatedTokens.Count > 0 ? generatedTokens[^1] : 0, generatedText, ref lastStreamedText);
                }
            }
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

    private MtmdBitmapWrapperHandle CreateImageBitmap(byte[] imageBytes)
    {
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImageBitmap: imageBytes.Length={imageBytes?.Length ?? -1}");
        var mtmd = EnsureMtmdContext();
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImageBitmap: mtmd.Handle=0x{mtmd.Handle.ToInt64():X}");
        var wrapper = MtmdNativeMethods.mtmd_helper_bitmap_init_from_buf(mtmd.Handle, imageBytes, (nuint)imageBytes.Length, false);
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImageBitmap: wrapper.bitmap=0x{wrapper.bitmap.ToInt64():X}");
        if (wrapper.bitmap == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to decode the supplied image bytes for Gemma 4 multimodal inference.");
        }

        return new MtmdBitmapWrapperHandle(wrapper);
    }

    private MtmdInputChunksHandle CreateImagePromptChunks(
        string prompt,
        IntPtr bitmapHandle,
        SeasonLlmGenerationOptions options,
        Utf8StringArena strings,
        out string promptText)
    {
        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] CreateImagePromptChunks: getting default marker");
        var marker = NativeMethods.PtrToString(MtmdNativeMethods.mtmd_default_marker());
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImagePromptChunks: marker='{marker}'");
        promptText = SeasonLlmModel.BuildGemma4SingleImagePrompt(prompt, marker);
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImagePromptChunks: promptText.Length={promptText?.Length ?? -1}");

        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] CreateImagePromptChunks: mtmd_input_chunks_init");
        var chunksHandle = MtmdNativeMethods.mtmd_input_chunks_init();
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImagePromptChunks: chunksHandle=0x{chunksHandle.ToInt64():X}");
        if (chunksHandle == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to allocate mtmd input chunks.");
        }

        var chunks = new MtmdInputChunksHandle(chunksHandle);
        var inputTextPtr = strings.Add(promptText);
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImagePromptChunks: input.text=0x{inputTextPtr.ToInt64():X}, add_special={options.AddSpecialTokens}, parse_special={options.ParseSpecialTokens}");
        var input = new MtmdNativeMethods.NativeMtmdInputText
        {
            text = inputTextPtr,
            add_special = NativeMethods.ToNativeBool(options.AddSpecialTokens),
            parse_special = NativeMethods.ToNativeBool(options.ParseSpecialTokens)
        };

        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImagePromptChunks: calling mtmd_tokenize, ctx=0x{EnsureMtmdContext().Handle.ToInt64():X}, bitmapHandle=0x{bitmapHandle.ToInt64():X}");
        var status = MtmdNativeMethods.mtmd_tokenize(EnsureMtmdContext().Handle, chunksHandle, ref input, [bitmapHandle], 1);
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] CreateImagePromptChunks: mtmd_tokenize returned status={status}");
        if (status == 0)
        {
            return chunks;
        }

        chunks.Dispose();
        throw status switch
        {
            1 => new InvalidOperationException("The Gemma 4 multimodal prompt marker count does not match the supplied image count."),
            2 => new InvalidOperationException("Gemma 4 multimodal image preprocessing failed inside mtmd."),
            _ => new InvalidOperationException($"mtmd_tokenize failed with native status {status}.")
        };
    }

    private void EvaluateImagePrompt(IntPtr chunksHandle, bool logitsLast, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var status = MtmdNativeMethods.mtmd_helper_eval_chunks(EnsureMtmdContext().Handle, _handle, chunksHandle, _position, 0, _batchSize, logitsLast, out var newPosition);
        if (status == 0)
        {
            _position = newPosition;
            return;
        }

        if (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }

        throw new InvalidOperationException($"mtmd_helper_eval_chunks failed with native status {status}.");
    }

    private MtmdContextHandle EnsureMtmdContext()
    {
        ThrowIfDisposed();
        if (_mtmdContext is not null)
        {
            System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] EnsureMtmdContext: already initialized");
            return _mtmdContext;
        }

        if (string.IsNullOrWhiteSpace(_model.MmprojPath))
        {
            throw new InvalidOperationException("This model was not initialized with MmprojPath, so image input is unavailable.");
        }

        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] EnsureMtmdContext: initializing mtmd, MmprojPath={_model.MmprojPath}");
        using var strings = new Utf8StringArena();
        var ctxParams = MtmdNativeMethods.mtmd_context_params_default();
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] EnsureMtmdContext: ctxParams_default returned, use_gpu={ctxParams.use_gpu}, n_threads={ctxParams.n_threads}");
        ctxParams.use_gpu = NativeMethods.ToNativeBool(_model.MmprojUseGpu);
        ctxParams.n_threads = _threadCount;
        ctxParams.flash_attn_type = _flashAttentionType;
        ctxParams.image_min_tokens = _model.ImageMinTokens;
        ctxParams.image_max_tokens = _model.ImageMaxTokens;
        ctxParams.warmup = 0;

        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] EnsureMtmdContext: calling mtmd_init_from_file");
        var handle = MtmdNativeMethods.mtmd_init_from_file(strings.Add(_model.MmprojPath), _model.Handle, ctxParams);
        System.Diagnostics.Debug.WriteLine($"[SeasonLLM][Gemma4] EnsureMtmdContext: mtmd_init_from_file returned 0x{handle.ToInt64():X}");
        if (handle == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to initialize mtmd with the configured mmproj file.");
        }

        var mtmd = new MtmdContextHandle(handle);
        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] EnsureMtmdContext: checking mtmd_support_vision");
        if (!MtmdNativeMethods.mtmd_support_vision(handle))
        {
            mtmd.Dispose();
            throw new InvalidOperationException("The configured mmproj/model pair does not expose vision input support.");
        }

        System.Diagnostics.Debug.WriteLine("[SeasonLLM][Gemma4] EnsureMtmdContext: mtmd initialized successfully");
        _mtmdContext = mtmd;
        return mtmd;
    }

    private void EnsureGemma4SingleImageSupport()
    {
        if (!_model.HasMmproj)
        {
            throw new InvalidOperationException("Gemma 4 single-image inference requires a model initialized with MmprojPath.");
        }

        var modelFileName = Path.GetFileName(_model.ModelPath);
        if (modelFileName.IndexOf("gemma-4-e4b-it", StringComparison.OrdinalIgnoreCase) < 0)
        {
            throw new NotSupportedException("The current multimodal wrapper only supports gemma-4-E4B-it + mmproj for single-image question answering.");
        }
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
            if (SeasonLlmGrammarHelpers.TryResolveGrammar(options, out var grammar, out var grammarRoot))
            {
                using var strings = new Utf8StringArena();
                var grammarSampler = NativeMethods.llama_sampler_init_grammar(
                    _model.VocabHandle,
                    strings.Add(grammar),
                    strings.Add(grammarRoot));

                if (grammarSampler == IntPtr.Zero)
                {
                    throw new InvalidOperationException("Failed to initialize the grammar sampler. Check the grammar or JsonSchema.");
                }

                NativeMethods.llama_sampler_chain_add(chain, grammarSampler);
            }

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

    private static string DecodeUtf8Bytes(Decoder decoder, byte[] bytes, bool flush)
    {
        ArgumentNullException.ThrowIfNull(decoder);

        if (bytes.Length == 0 && !flush)
        {
            return string.Empty;
        }

        var charBuffer = new char[Encoding.UTF8.GetMaxCharCount(Math.Max(bytes.Length, 1))];
        var charCount = decoder.GetChars(bytes, 0, bytes.Length, charBuffer, 0, flush);
        return charCount == 0 ? string.Empty : new string(charBuffer, 0, charCount);
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

        _mtmdContext?.Dispose();
        _mtmdContext = null;

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

        _mtmdContext?.Dispose();
        _mtmdContext = null;
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
