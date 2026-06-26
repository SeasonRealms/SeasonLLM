// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace SeasonLLM;

public static class SeasonLLM
{
    private static readonly object s_initLock = new();

    private static bool s_backendInitialized;
    private static NativeMethods.GgmlLogCallback? s_logThunk;
    private static Action<SeasonLlmLogLevel, string>? s_logCallback;

    public static bool IsSupported => OperatingSystem.IsWindows();

    public static string SystemInfo
    {
        get
        {
            EnsureInitialized();
            return NativeMethods.PtrToString(NativeMethods.llama_print_system_info());
        }
    }

    public static void EnsureInitialized()
    {
        EnsureSupported();

        if (s_backendInitialized)
        {
            return;
        }

        lock (s_initLock)
        {
            if (s_backendInitialized)
            {
                return;
            }

            NativeMethods.llama_backend_init();
            s_backendInitialized = true;
        }
    }

    public static void SetLogCallback(Action<SeasonLlmLogLevel, string>? callback)
    {
        EnsureInitialized();
        s_logCallback = callback;

        if (callback is null)
        {
            s_logThunk = null;
            NativeMethods.llama_log_set(null, IntPtr.Zero);
            return;
        }

        s_logThunk = static (level, text, _) =>
        {
            var managed = s_logCallback;
            if (managed is null)
            {
                return;
            }

            managed((SeasonLlmLogLevel)level, NativeMethods.PtrToString(text));
        };

        NativeMethods.llama_log_set(s_logThunk, IntPtr.Zero);
    }

    public static SeasonLlmModel CreateModel(SeasonLlmModelOptions options)
    {
        EnsureInitialized();
        ArgumentNullException.ThrowIfNull(options);
        return new SeasonLlmModel(options);
    }

    public static IReadOnlyList<string> GetAvailableBackends()
    {
        EnsureInitialized();
        return global::SeasonGGML.GGML.GetAvailableBackends()
            .Select(static backend => backend.Name)
            .ToArray();
    }

    internal static void EnsureSupported()
    {
        if (!OperatingSystem.IsWindows())
        {
            throw new PlatformNotSupportedException(
                "SeasonLLM currently ships llama.cpp native binaries only for Windows.");
        }
    }
}
