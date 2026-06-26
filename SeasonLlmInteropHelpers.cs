// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

using System.Globalization;

namespace SeasonLLM;

internal sealed class Utf8StringArena : IDisposable
{
    private readonly List<IntPtr> _buffers = [];

    public IntPtr Add(string? value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return IntPtr.Zero;
        }

        var ptr = Marshal.StringToCoTaskMemUTF8(value);
        _buffers.Add(ptr);
        return ptr;
    }

    public void Dispose()
    {
        foreach (var ptr in _buffers)
        {
            if (ptr != IntPtr.Zero)
            {
                Marshal.FreeCoTaskMem(ptr);
            }
        }

        _buffers.Clear();
    }
}

internal sealed class PinnedChatMessages : IDisposable
{
    private readonly NativeMethods.NativeLlamaChatMessage[] _messages;
    private GCHandle _handle;
    private bool _disposed;

    private PinnedChatMessages(NativeMethods.NativeLlamaChatMessage[] messages)
    {
        _messages = messages;
        _handle = GCHandle.Alloc(_messages, GCHandleType.Pinned);
    }

    public int Count => _messages.Length;
    public IntPtr Pointer => _handle.AddrOfPinnedObject();

    public static PinnedChatMessages? Create(IReadOnlyList<SeasonLlmChatMessage>? messages, Utf8StringArena strings)
    {
        if (messages is null || messages.Count == 0)
        {
            return null;
        }

        var native = new NativeMethods.NativeLlamaChatMessage[messages.Count];
        for (var i = 0; i < messages.Count; i++)
        {
            var message = messages[i] ?? throw new ArgumentException("Chat messages cannot contain null values.", nameof(messages));
            native[i] = new NativeMethods.NativeLlamaChatMessage
            {
                role = strings.Add(message.Role),
                content = strings.Add(message.Content)
            };
        }

        return new PinnedChatMessages(native);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_handle.IsAllocated)
        {
            _handle.Free();
        }
    }
}

internal sealed unsafe class LlamaBatchBuffer : IDisposable
{
    private NativeMethods.NativeLlamaBatch _batch;
    private readonly int _capacity;
    private bool _disposed;

    public LlamaBatchBuffer(int capacity, int maxSequences)
    {
        if (capacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity));
        }

        if (maxSequences <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSequences));
        }

        _capacity = capacity;
        _batch = NativeMethods.llama_batch_init(capacity, 0, maxSequences);

        if (_batch.token == IntPtr.Zero || _batch.pos == IntPtr.Zero || _batch.n_seq_id == IntPtr.Zero ||
            _batch.seq_id == IntPtr.Zero || _batch.logits == IntPtr.Zero)
        {
            Dispose();
            throw new InvalidOperationException("Failed to allocate llama batch buffers.");
        }
    }

    public NativeMethods.NativeLlamaBatch Batch => _batch;

    public void SetTokens(ReadOnlySpan<int> tokens, int positionBase, bool requestLogitsOnLastToken, int sequenceId = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (tokens.Length > _capacity)
        {
            throw new ArgumentOutOfRangeException(nameof(tokens));
        }

        var tokenPtr = (int*)_batch.token;
        var posPtr = (int*)_batch.pos;
        var nSeqIdPtr = (int*)_batch.n_seq_id;
        var seqIdPtr = (int**)_batch.seq_id;
        var logitsPtr = (sbyte*)_batch.logits;

        for (var i = 0; i < tokens.Length; i++)
        {
            tokenPtr[i] = tokens[i];
            posPtr[i] = positionBase + i;
            nSeqIdPtr[i] = 1;
            seqIdPtr[i][0] = sequenceId;
            logitsPtr[i] = (sbyte)(requestLogitsOnLastToken && i == tokens.Length - 1 ? 1 : 0);
        }

        _batch.n_tokens = tokens.Length;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        NativeMethods.llama_batch_free(_batch);
        _batch = default;
    }
}

internal sealed class CancellationBridge : IDisposable
{
    private static readonly NativeMethods.GgmlAbortCallback s_abortThunk = static data =>
    {
        if (data == IntPtr.Zero)
        {
            return false;
        }

        var handle = GCHandle.FromIntPtr(data);
        return handle.Target is CancellationToken token && token.IsCancellationRequested;
    };

    private GCHandle _tokenHandle;
    private bool _disposed;

    private CancellationBridge(CancellationToken token)
    {
        _tokenHandle = GCHandle.Alloc(token);
    }

    public IntPtr UserData => GCHandle.ToIntPtr(_tokenHandle);
    public static NativeMethods.GgmlAbortCallback AbortThunk => s_abortThunk;

    public static CancellationBridge? Create(CancellationToken token)
    {
        return token.CanBeCanceled ? new CancellationBridge(token) : null;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_tokenHandle.IsAllocated)
        {
            _tokenHandle.Free();
        }
    }
}
