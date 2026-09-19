// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

public static class SeasonLLM
{
    private static readonly object s_initLock = new();

    private static bool s_backendInitialized;
    private static NativeMethods.GgmlLogCallback? s_logThunk;
    private static Action<SeasonLlmLogLevel, string>? s_logCallback;

    public static bool IsSupported =>
        OperatingSystem.IsWindows() ||
        (OperatingSystem.IsMacCatalyst() && RuntimeInformation.ProcessArchitecture == Architecture.Arm64);

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

        s_logThunk = LogThunk;
        NativeMethods.llama_log_set(s_logThunk, IntPtr.Zero);
    }

    // The native side keeps this function pointer for the process lifetime, so the thunk
    // must be a static method with MonoPInvokeCallback: the AOT compiler then pre-generates
    // its native-to-managed wrapper. A lambda (or any other unattributed method) would need
    // that wrapper JIT-compiled on first use, which aborts aot-only Release builds with
    // "Attempting to JIT compile method '(wrapper native-to-managed) ...'".
#if IOS || MACCATALYST
    [ObjCRuntime.MonoPInvokeCallback(typeof(NativeMethods.GgmlLogCallback))]
#endif
    private static void LogThunk(int level, IntPtr text, IntPtr userData)
    {
        var managed = s_logCallback;
        if (managed is null)
        {
            return;
        }

        managed((SeasonLlmLogLevel)level, NativeMethods.PtrToString(text));
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
        return global::Season.GGML.GGML.GetAvailableBackends()
            .Select(static backend => backend.Name)
            .ToArray();
    }

    internal static void EnsureSupported()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsMacCatalyst())
        {
            throw new PlatformNotSupportedException(
                "SeasonLLM currently ships llama.cpp native binaries only for Windows and Mac Catalyst (Apple Silicon).");
        }

        // The Mac Catalyst artifacts are pure arm64 slices, so on an Intel Mac - or under
        // Rosetta - the dylibs cannot even be opened. Report that reason instead of
        // letting the first P/Invoke surface a bare DllNotFoundException.
        if (OperatingSystem.IsMacCatalyst() && RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
        {
            throw new PlatformNotSupportedException(
                "SeasonLLM ships Mac Catalyst llama.cpp native binaries for Apple Silicon (arm64) only; " +
                $"this process runs as {RuntimeInformation.ProcessArchitecture}.");
        }
    }
}
