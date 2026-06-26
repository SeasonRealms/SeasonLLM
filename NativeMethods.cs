// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace SeasonLLM;

internal static class NativeMethods
{
    internal const string LibraryName = "llama";
    internal const int LlamaTokenNull = -1;

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeLlamaModelParams
    {
        public IntPtr devices;
        public IntPtr tensor_buft_overrides;
        public int n_gpu_layers;
        public int split_mode;
        public int main_gpu;
        public IntPtr tensor_split;
        public IntPtr progress_callback;
        public IntPtr progress_callback_user_data;
        public IntPtr kv_overrides;
        public byte vocab_only;
        public byte use_mmap;
        public byte use_direct_io;
        public byte use_mlock;
        public byte check_tensors;
        public byte use_extra_bufts;
        public byte no_host;
        public byte no_alloc;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeLlamaContextParams
    {
        public uint n_ctx;
        public uint n_batch;
        public uint n_ubatch;
        public uint n_seq_max;
        public uint n_rs_seq;
        public uint n_outputs_max;
        public int n_threads;
        public int n_threads_batch;
        public int ctx_type;
        public int rope_scaling_type;
        public int pooling_type;
        public int attention_type;
        public int flash_attn_type;
        public float rope_freq_base;
        public float rope_freq_scale;
        public float yarn_ext_factor;
        public float yarn_attn_factor;
        public float yarn_beta_fast;
        public float yarn_beta_slow;
        public uint yarn_orig_ctx;
        public float defrag_thold;
        public IntPtr cb_eval;
        public IntPtr cb_eval_user_data;
        public int type_k;
        public int type_v;
        public IntPtr abort_callback;
        public IntPtr abort_callback_data;
        public byte embeddings;
        public byte offload_kqv;
        public byte no_perf;
        public byte op_offload;
        public byte swa_full;
        public byte kv_unified;
        public IntPtr samplers;
        public nuint n_samplers;
        public IntPtr ctx_other;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeLlamaSamplerChainParams
    {
        public byte no_perf;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeLlamaChatMessage
    {
        public IntPtr role;
        public IntPtr content;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeLlamaBatch
    {
        public int n_tokens;
        public IntPtr token;
        public IntPtr embd;
        public IntPtr pos;
        public IntPtr n_seq_id;
        public IntPtr seq_id;
        public IntPtr logits;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate bool NativeProgressCallback(float progress, IntPtr userData);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate bool GgmlAbortCallback(IntPtr data);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void GgmlLogCallback(int level, IntPtr text, IntPtr userData);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern NativeLlamaModelParams llama_model_default_params();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern NativeLlamaContextParams llama_context_default_params();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern NativeLlamaSamplerChainParams llama_sampler_chain_default_params();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_backend_init();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_backend_free();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_model_load_from_file(IntPtr pathModel, NativeLlamaModelParams @params);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_model_free(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_init_from_model(IntPtr model, NativeLlamaContextParams @params);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_free(IntPtr ctx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern uint llama_n_ctx(IntPtr ctx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern uint llama_n_batch(IntPtr ctx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern uint llama_n_seq_max(IntPtr ctx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_get_memory(IntPtr ctx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_memory_clear(IntPtr memory, [MarshalAs(UnmanagedType.I1)] bool data);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_model_get_vocab(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_model_n_ctx_train(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_model_n_embd(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern ulong llama_model_n_params(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern ulong llama_model_size(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_model_desc(IntPtr model, byte[] buffer, nuint bufferSize);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool llama_model_has_encoder(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool llama_model_has_decoder(IntPtr model);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_model_chat_template(IntPtr model, IntPtr name);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_vocab_n_tokens(IntPtr vocab);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.I1)]
    internal static extern bool llama_vocab_is_eog(IntPtr vocab, int token);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_tokenize(
        IntPtr vocab,
        IntPtr text,
        int textLen,
        [Out] int[] tokens,
        int nTokensMax,
        [MarshalAs(UnmanagedType.I1)] bool addSpecial,
        [MarshalAs(UnmanagedType.I1)] bool parseSpecial);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_detokenize(
        IntPtr vocab,
        int[] tokens,
        int tokenCount,
        [Out] byte[] text,
        int textLenMax,
        [MarshalAs(UnmanagedType.I1)] bool removeSpecial,
        [MarshalAs(UnmanagedType.I1)] bool unparseSpecial);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_token_to_piece(
        IntPtr vocab,
        int token,
        [Out] byte[] buffer,
        int length,
        int lstrip,
        [MarshalAs(UnmanagedType.I1)] bool special);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_chat_apply_template(
        IntPtr tmpl,
        IntPtr chat,
        nuint nMsg,
        [MarshalAs(UnmanagedType.I1)] bool addAssistant,
        [Out] byte[] buffer,
        int length);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern NativeLlamaBatch llama_batch_init(int nTokens, int embd, int nSeqMax);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_batch_free(NativeLlamaBatch batch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_decode(IntPtr ctx, NativeLlamaBatch batch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_set_n_threads(IntPtr ctx, int nThreads, int nThreadsBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_set_abort_callback(IntPtr ctx, GgmlAbortCallback? abortCallback, IntPtr abortCallbackData);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_chain_init(NativeLlamaSamplerChainParams @params);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_sampler_chain_add(IntPtr chain, IntPtr sampler);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_init_greedy();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_init_dist(uint seed);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_init_top_k(int k);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_init_top_p(float p, nuint minKeep);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_init_min_p(float p, nuint minKeep);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_init_temp(float temperature);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_sampler_init_penalties(int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern int llama_sampler_sample(IntPtr sampler, IntPtr ctx, int idx);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_sampler_accept(IntPtr sampler, int token);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_sampler_free(IntPtr sampler);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr llama_print_system_info();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void llama_log_set(GgmlLogCallback? logCallback, IntPtr userData);

    internal static byte ToNativeBool(bool value) => value ? (byte)1 : (byte)0;

    internal static string PtrToString(IntPtr ptr)
    {
        return ptr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
    }
}
