// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

#pragma warning disable CA2255

using System.Reflection;
using System.Runtime.CompilerServices;

namespace Season.LLM;

/// <summary>
/// Resolves the llama.cpp shared objects on Linux desktop.
///
/// ld.so resolves DT_NEEDED entries from the loader search path only - it never
/// looks next to the library being opened - so opening libllama.so by name fails
/// unless libggml.so.0 / libggml-base.so.0 are already in the process. The resolver
/// therefore dlopen's the chain in dependency order (ggml-base, ggml, then
/// llama/mtmd) by absolute path; glibc registers each SONAME on load, which then
/// satisfies the next library's NEEDED entries. Windows resolves through the DLL
/// search path (the MSBuild Content items copy the .dll files to the output root)
/// and iOS links everything statically into "__Internal", so only Linux needs this.
/// </summary>
internal static class NativeLibraryResolver
{
    [ModuleInitializer]
    public static void Init()
    {
        if (OperatingSystem.IsLinux())
        {
            NativeLibrary.SetDllImportResolver(typeof(NativeLibraryResolver).Assembly, Resolve);
        }
    }

    private static readonly object s_lock = new();
    private static readonly Dictionary<string, IntPtr> s_handles = new(StringComparer.Ordinal);

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != NativeMethods.LibraryName && libraryName != MtmdNativeMethods.LibraryName)
        {
            return IntPtr.Zero;
        }

        lock (s_lock)
        {
            if (s_handles.TryGetValue(libraryName, out var cached) && cached != IntPtr.Zero)
            {
                return cached;
            }

            var directories = EnumerateNativeSearchDirectories();

            // Dependencies first, in NEEDED order. Preloading is idempotent (dlopen
            // returns the existing mapping), so repeating it per library is fine.
            // libmtmd.so additionally NEEDs libllama.so.0.
            Preload(directories, "libggml-base.so", assembly, searchPath);
            Preload(directories, "libggml.so", assembly, searchPath);
            if (libraryName == MtmdNativeMethods.LibraryName)
            {
                Preload(directories, "libllama.so", assembly, searchPath);
            }

            var fileName = libraryName == NativeMethods.LibraryName ? "libllama.so" : "libmtmd.so";
            foreach (var directory in directories)
            {
                var candidate = Path.Combine(directory, fileName);
                if (File.Exists(candidate) &&
                    NativeLibrary.TryLoad(candidate, assembly, searchPath, out var handle))
                {
                    s_handles[libraryName] = handle;
                    return handle;
                }
            }
        }

        // Fall through to the default .NET probing, which covers a host that placed
        // the libraries on the system loader search path itself.
        return IntPtr.Zero;
    }

    private static void Preload(string[] directories, string fileName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        foreach (var directory in directories)
        {
            var candidate = Path.Combine(directory, fileName);
            if (File.Exists(candidate))
            {
                NativeLibrary.TryLoad(candidate, assembly, searchPath, out _);
                return;
            }
        }
    }

    /// <summary>
    /// The two layouts a consumer sees: the flat output root (<c>Content</c> items
    /// with <c>Link</c> flatten the RID folders) and the RID layout preserved inside
    /// the NuGet package.
    /// </summary>
    private static string[] EnumerateNativeSearchDirectories()
    {
        var baseDirectory = Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory);
        var directories = new List<string>(2) { baseDirectory };

        var arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            _ => null,
        };
        if (arch is not null)
        {
            directories.Add(Path.Combine(baseDirectory, "runtimes", $"linux-{arch}", "native"));
        }

        return directories.ToArray();
    }
}

#pragma warning restore CA2255
