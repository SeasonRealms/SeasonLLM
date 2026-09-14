// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

internal static class MtmdNativeMethods
{
    internal const string LibraryName = "mtmd";

    internal const int MtmdInputChunkTypeText = 0;
    internal const int MtmdInputChunkTypeImage = 1;

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeMtmdContextParams
    {
        public byte use_gpu;
        public byte print_timings;
        public int n_threads;
        public IntPtr image_marker;
        public IntPtr media_marker;
        public int flash_attn_type;
        public byte warmup;
        public int image_min_tokens;
        public int image_max_tokens;
        public IntPtr cb_eval;
        public IntPtr cb_eval_user_data;
        public int batch_max_tokens;
        public IntPtr progress_callback;
        public IntPtr progress_callback_user_data;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeMtmdInputText
    {
        public IntPtr text;
        public byte add_special;
        public byte parse_special;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeMtmdBitmapWrapper
    {
        public IntPtr bitmap;
        public IntPtr video_ctx;
    }

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr mtmd_default_marker();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern NativeMtmdContextParams mtmd_context_params_default();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr mtmd_init_from_file(IntPtr mmprojFileName, IntPtr textModel, NativeMtmdContextParams ctxParams);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void mtmd_free(IntPtr ctx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool mtmd_support_vision(IntPtr ctx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern NativeMtmdBitmapWrapper mtmd_helper_bitmap_init_from_buf(IntPtr ctx, byte[] buf, nuint len, [MarshalAs(UnmanagedType.I1)] bool placeholder);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr mtmd_input_chunks_init();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void mtmd_input_chunks_free(IntPtr chunks);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int mtmd_tokenize(IntPtr ctx, IntPtr output, ref NativeMtmdInputText text, IntPtr[] bitmaps, nuint nBitmaps);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint mtmd_helper_get_n_tokens(IntPtr chunks);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int mtmd_helper_eval_chunks(IntPtr ctx, IntPtr lctx, IntPtr chunks, int nPast, int seqId, int nBatch, [MarshalAs(UnmanagedType.I1)] bool logitsLast, out int newNPast);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void mtmd_bitmap_free(IntPtr bitmap);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void mtmd_helper_video_free(IntPtr ctx);
}

internal sealed class MtmdContextHandle : IDisposable
{
    public IntPtr Handle { get; private set; }

    public MtmdContextHandle(IntPtr handle)
    {
        Handle = handle;
    }

    public void Dispose()
    {
        if (Handle != IntPtr.Zero)
        {
            MtmdNativeMethods.mtmd_free(Handle);
            Handle = IntPtr.Zero;
        }
    }
}

internal sealed class MtmdInputChunksHandle : IDisposable
{
    public IntPtr Handle { get; private set; }

    public MtmdInputChunksHandle(IntPtr handle)
    {
        Handle = handle;
    }

    public void Dispose()
    {
        if (Handle != IntPtr.Zero)
        {
            MtmdNativeMethods.mtmd_input_chunks_free(Handle);
            Handle = IntPtr.Zero;
        }
    }
}

internal sealed class MtmdBitmapWrapperHandle : IDisposable
{
    private IntPtr _videoContext;

    public IntPtr BitmapHandle { get; private set; }

    public MtmdBitmapWrapperHandle(MtmdNativeMethods.NativeMtmdBitmapWrapper wrapper)
    {
        BitmapHandle = wrapper.bitmap;
        _videoContext = wrapper.video_ctx;
    }

    public void Dispose()
    {
        if (BitmapHandle != IntPtr.Zero)
        {
            MtmdNativeMethods.mtmd_bitmap_free(BitmapHandle);
            BitmapHandle = IntPtr.Zero;
        }

        if (_videoContext != IntPtr.Zero)
        {
            MtmdNativeMethods.mtmd_helper_video_free(_videoContext);
            _videoContext = IntPtr.Zero;
        }
    }
}
