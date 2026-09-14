// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

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
    // ==== TEMPORARY DIAGNOSTICS: trace abort-callback behavior (remove after debugging) ====
    private static long s_thunkCalls;
    private static IntPtr s_lastThunkData;
    private static int s_bridgeSeq;

    private static readonly NativeMethods.GgmlAbortCallback s_abortThunk = static data =>
    {
        if (data == IntPtr.Zero)
        {
            return (byte)0;
        }

        object? target;
        try
        {
            target = GCHandle.FromIntPtr(data).Target;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine(
                $"[SeasonLLM][AbortThunk] INVALID GCHandle data=0x{data.ToInt64():X} ex={ex.GetType().Name}: {ex.Message}");
            return (byte)0;
        }

        var token = target as CancellationToken?;
        var canceled = token?.IsCancellationRequested == true;

        var call = System.Threading.Interlocked.Increment(ref s_thunkCalls);
        if (call == 1 || data != s_lastThunkData || canceled || token is null)
        {
            s_lastThunkData = data;
            System.Diagnostics.Debug.WriteLine(
                $"[SeasonLLM][AbortThunk] call#{call} data=0x{data.ToInt64():X} targetType={(target?.GetType().Name ?? "null")} canceled={canceled} tokenHash={(token is CancellationToken t ? t.GetHashCode().ToString() : "n/a")}");
        }

        return (byte)(canceled ? 1 : 0);
    };

    private GCHandle _tokenHandle;
    private bool _disposed;
    private readonly int _seq;

    private CancellationBridge(CancellationToken token)
    {
        _tokenHandle = GCHandle.Alloc(token);
        _seq = System.Threading.Interlocked.Increment(ref s_bridgeSeq);
        System.Diagnostics.Debug.WriteLine(
            $"[SeasonLLM][AbortThunk] bridge#{_seq} CREATED data=0x{GCHandle.ToIntPtr(_tokenHandle).ToInt64():X} canBeCanceled={token.CanBeCanceled} alreadyCanceled={token.IsCancellationRequested} tokenHash={token.GetHashCode()}");
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
        System.Diagnostics.Debug.WriteLine(
            $"[SeasonLLM][AbortThunk] bridge#{_seq} DISPOSED data=0x{GCHandle.ToIntPtr(_tokenHandle).ToInt64():X}");
        if (_tokenHandle.IsAllocated)
        {
            _tokenHandle.Free();
        }
    }
}
