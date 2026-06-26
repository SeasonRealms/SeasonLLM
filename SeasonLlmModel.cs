// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace SeasonLLM;

public sealed class SeasonLlmModel : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;
    private int _activeContexts;

    internal SeasonLlmModel(SeasonLlmModelOptions options)
    {
        SeasonLLM.EnsureSupported();
        ArgumentNullException.ThrowIfNull(options);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.ModelPath);

        using var strings = new Utf8StringArena();
        using var backendDevices = global::SeasonGGML.GGML.CreateBackendSelection(options.Backend, options.ParamsBackend);

        var native = NativeMethods.llama_model_default_params();
        native.devices = backendDevices?.Pointer ?? IntPtr.Zero;
        native.n_gpu_layers = options.GpuLayers;
        native.main_gpu = options.MainGpu;
        native.use_mmap = NativeMethods.ToNativeBool(options.UseMmap);
        native.use_mlock = NativeMethods.ToNativeBool(options.UseMlock);
        native.use_direct_io = NativeMethods.ToNativeBool(options.UseDirectIo);
        native.check_tensors = NativeMethods.ToNativeBool(options.CheckTensors);
        native.vocab_only = NativeMethods.ToNativeBool(options.VocabularyOnly);

        NativeMethods.NativeProgressCallback? progressThunk = null;
        GCHandle progressHandle = default;

        if (options.LoadProgressCallback is not null)
        {
            progressHandle = GCHandle.Alloc(options.LoadProgressCallback);
            progressThunk = static (progress, userData) =>
            {
                if (userData == IntPtr.Zero)
                {
                    return true;
                }

                var callback = GCHandle.FromIntPtr(userData).Target as Action<float>;
                callback?.Invoke(progress);
                return true;
            };

            native.progress_callback = Marshal.GetFunctionPointerForDelegate(progressThunk);
            native.progress_callback_user_data = GCHandle.ToIntPtr(progressHandle);
        }

        try
        {
            _handle = NativeMethods.llama_model_load_from_file(strings.Add(options.ModelPath), native);
        }
        finally
        {
            if (progressHandle.IsAllocated)
            {
                progressHandle.Free();
            }

            GC.KeepAlive(progressThunk);
        }

        if (_handle == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                "Failed to load llama model. Check the GGUF path and native runtime dependencies.");
        }

        ModelPath = options.ModelPath;
        Backend = options.Backend;
        ParamsBackend = options.ParamsBackend;
        ResolvedBackends = backendDevices?.Names.ToArray() ?? [];
    }

    public string ModelPath { get; }
    public string? Backend { get; }
    public string? ParamsBackend { get; }
    public IReadOnlyList<string> ResolvedBackends { get; }

    internal IntPtr Handle
    {
        get
        {
            ThrowIfDisposed();
            return _handle;
        }
    }

    internal IntPtr VocabHandle
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_model_get_vocab(_handle);
        }
    }

    public string Description
    {
        get
        {
            ThrowIfDisposed();
            return ReadModelString(buffer => NativeMethods.llama_model_desc(_handle, buffer, (nuint)buffer.Length));
        }
    }

    public ulong ParameterCount
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_model_n_params(_handle);
        }
    }

    public ulong SizeInBytes
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_model_size(_handle);
        }
    }

    public int TrainingContextSize
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_model_n_ctx_train(_handle);
        }
    }

    public int EmbeddingSize
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_model_n_embd(_handle);
        }
    }

    public bool HasEncoder
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_model_has_encoder(_handle);
        }
    }

    public bool HasDecoder
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_model_has_decoder(_handle);
        }
    }

    public string DefaultChatTemplate
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.PtrToString(NativeMethods.llama_model_chat_template(_handle, IntPtr.Zero));
        }
    }

    public int VocabularySize
    {
        get
        {
            ThrowIfDisposed();
            return NativeMethods.llama_vocab_n_tokens(VocabHandle);
        }
    }

    public SeasonLlmContext CreateContext(SeasonLlmContextOptions? options = null)
    {
        ThrowIfDisposed();
        return new SeasonLlmContext(this, options ?? new SeasonLlmContextOptions());
    }

    public int[] Tokenize(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(text);

        using var strings = new Utf8StringArena();
        var vocab = VocabHandle;
        var textPtr = strings.Add(text);
        var utf8Length = Encoding.UTF8.GetByteCount(text);
        var tokens = new int[Math.Max(utf8Length + 8, 32)];

        var count = NativeMethods.llama_tokenize(vocab, textPtr, utf8Length, tokens, tokens.Length, addSpecial, parseSpecial);
        if (count == int.MinValue)
        {
            throw new InvalidOperationException("Tokenization overflowed the native int32 token count limit.");
        }

        if (count < 0)
        {
            tokens = new int[-count];
            count = NativeMethods.llama_tokenize(vocab, textPtr, utf8Length, tokens, tokens.Length, addSpecial, parseSpecial);
        }

        if (count < 0)
        {
            throw new InvalidOperationException("llama_tokenize failed.");
        }

        return tokens.AsSpan(0, count).ToArray();
    }

    public string Detokenize(IReadOnlyList<int> tokens, bool removeSpecial = false, bool unparseSpecial = false)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(tokens);

        if (tokens.Count == 0)
        {
            return string.Empty;
        }

        var tokenArray = tokens as int[] ?? tokens.ToArray();
        var buffer = new byte[Math.Max(tokenArray.Length * 8, 64)];
        var count = NativeMethods.llama_detokenize(VocabHandle, tokenArray, tokenArray.Length, buffer, buffer.Length, removeSpecial, unparseSpecial);

        if (count < 0)
        {
            buffer = new byte[-count];
            count = NativeMethods.llama_detokenize(VocabHandle, tokenArray, tokenArray.Length, buffer, buffer.Length, removeSpecial, unparseSpecial);
        }

        if (count < 0)
        {
            var builder = new StringBuilder();
            foreach (var token in tokenArray)
            {
                builder.Append(TokenToPiece(token, unparseSpecial));
            }

            return builder.ToString();
        }

        return Encoding.UTF8.GetString(buffer, 0, count);
    }

    public string TokenToPiece(int token, bool special = false)
    {
        ThrowIfDisposed();

        var buffer = new byte[256];
        var count = NativeMethods.llama_token_to_piece(VocabHandle, token, buffer, buffer.Length, 0, special);
        if (count < 0)
        {
            buffer = new byte[-count];
            count = NativeMethods.llama_token_to_piece(VocabHandle, token, buffer, buffer.Length, 0, special);
        }

        if (count < 0)
        {
            throw new InvalidOperationException($"Failed to convert token {token} to text.");
        }

        return Encoding.UTF8.GetString(buffer, 0, count);
    }

    public string ApplyChatTemplate(
        IReadOnlyList<SeasonLlmChatMessage> messages,
        bool addAssistantGenerationPrompt = true,
        string? chatTemplate = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(messages);

        if (messages.Count == 0)
        {
            return addAssistantGenerationPrompt ? "assistant: " : string.Empty;
        }

        using var strings = new Utf8StringArena();
        using var nativeMessages = PinnedChatMessages.Create(messages, strings);

        var effectiveTemplate = string.IsNullOrWhiteSpace(chatTemplate) ? DefaultChatTemplate : chatTemplate;
        if (string.IsNullOrWhiteSpace(effectiveTemplate) || nativeMessages is null)
        {
            return BuildFallbackChatPrompt(messages, addAssistantGenerationPrompt);
        }

        var templatePtr = strings.Add(effectiveTemplate);
        var estimatedLength = Math.Max(256, messages.Sum(static message => message.Content.Length + message.Role.Length + 16) * 4);
        var buffer = new byte[estimatedLength];
        var count = NativeMethods.llama_chat_apply_template(templatePtr, nativeMessages.Pointer, (nuint)nativeMessages.Count, addAssistantGenerationPrompt, buffer, buffer.Length);

        if (count < 0)
        {
            throw new InvalidOperationException("llama_chat_apply_template failed.");
        }

        if (count >= buffer.Length)
        {
            buffer = new byte[count + 1];
            count = NativeMethods.llama_chat_apply_template(templatePtr, nativeMessages.Pointer, (nuint)nativeMessages.Count, addAssistantGenerationPrompt, buffer, buffer.Length);
            if (count < 0)
            {
                throw new InvalidOperationException("llama_chat_apply_template failed.");
            }
        }

        return Encoding.UTF8.GetString(buffer, 0, count);
    }

    internal void RentContext()
    {
        ThrowIfDisposed();
        Interlocked.Increment(ref _activeContexts);
    }

    internal void ReturnContext()
    {
        Interlocked.Decrement(ref _activeContexts);
    }

    internal static string BuildFallbackChatPrompt(IReadOnlyList<SeasonLlmChatMessage> messages, bool addAssistantGenerationPrompt)
    {
        var builder = new StringBuilder();
        foreach (var message in messages)
        {
            builder.Append(message.Role);
            builder.Append(": ");
            builder.AppendLine(message.Content);
        }

        if (addAssistantGenerationPrompt)
        {
            builder.Append("assistant: ");
        }

        return builder.ToString();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        if (Volatile.Read(ref _activeContexts) > 0)
        {
            throw new InvalidOperationException("Cannot dispose a SeasonLlmModel while one or more contexts are still alive.");
        }

        _disposed = true;

        if (_handle != IntPtr.Zero)
        {
            NativeMethods.llama_model_free(_handle);
            _handle = IntPtr.Zero;
        }

        GC.SuppressFinalize(this);
    }

    ~SeasonLlmModel()
    {
        if (_handle != IntPtr.Zero)
        {
            NativeMethods.llama_model_free(_handle);
            _handle = IntPtr.Zero;
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    private string ReadModelString(Func<byte[], int> nativeCall)
    {
        var buffer = new byte[512];
        while (true)
        {
            var count = nativeCall(buffer);
            if (count < 0)
            {
                return string.Empty;
            }

            if (count < buffer.Length)
            {
                return Encoding.UTF8.GetString(buffer, 0, count);
            }

            buffer = new byte[count + 1];
        }
    }
}
