using System.Buffers.Binary;
using System.Text.Json;
using ValveConvertTool.Configuration;
using ValveConvertTool.Core;
using ValveConvertTool.Discovery;
using ValveConvertTool.Infrastructure;

namespace ValveConvertTool.Pipeline;

internal sealed class ConversionPipeline(
    AppOptions options,
    AssetScanner scanner,
    FileSystemService fileSystem,
    BlenderRunner blender)
{
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    public async Task RunAsync(CancellationToken cancellationToken = default)
    {
        FileSystemService.ValidateDirectories(options.InputDirectory, options.OutputDirectory);

        var inputRoot = Path.GetFullPath(options.InputDirectory);
        var outputRoot = Path.GetFullPath(options.OutputDirectory);
        ConsoleReporter.Progress("OUTPUT", $"Preparing {outputRoot}");
        Directory.CreateDirectory(outputRoot);
        ConsoleReporter.Complete();
        if (inputRoot.StartsWith(outputRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                "The input directory cannot be inside the output directory; output cleanup would be unsafe.");

        var assets = scanner.Scan(inputRoot);
        var jobs = new List<AssetFileJob>();
        var results = new List<ConversionResult>();
        var reportPath = Path.Combine(outputRoot, "conversion-report.json");

        var expectedOutputs = BuildExpectedOutputs(assets, outputRoot);
        CleanOutputDirectory(outputRoot, reportPath, expectedOutputs);

        var processedAssetCount = 0;
        var audioAssetCount = assets.Count(asset => asset.Type == AssetType.Audio);
        var processedAudioCount = 0;
        var copiedAudioCount = 0;
        var lastCheckProgress = DateTime.MinValue;

        foreach (var asset in assets)
        {
            cancellationToken.ThrowIfCancellationRequested();

            processedAssetCount++;

            var now = DateTime.UtcNow;
            if (processedAssetCount == 1 || now - lastCheckProgress >= TimeSpan.FromSeconds(5))
            {
                ConsoleReporter.Progress(
                    "CHECK",
                    $"Checking {processedAssetCount:N0}/{assets.Count:N0} files; " +
                    $"audio {processedAudioCount:N0}/{audioAssetCount:N0} processed, " +
                    $"{copiedAudioCount:N0} copied");

                lastCheckProgress = now;
            }

            if (TryGetConversion(asset, out var jobType, out var outputExtension))
            {
                var outputRelative = Path.ChangeExtension(asset.RelativePath, outputExtension!);
                var outputPath = FileSystemService.GetOutputPath(outputRoot, outputRelative);

                if (asset.Type == AssetType.Texture && IsValidPng(outputPath))
                {
                    results.Add(new ConversionResult(
                        asset.RelativePath,
                        [outputPath],
                        ConversionStatus.Skipped,
                        "Valid output already exists; conversion skipped."));
                    continue;
                }
                if (asset.Type == AssetType.Model && IsValidGlb(outputPath))
                {
                    results.Add(new ConversionResult(
                        asset.RelativePath,
                        [outputPath],
                        ConversionStatus.Skipped,
                        "Valid self-contained GLB already exists; conversion skipped."));
                    continue;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                jobs.Add(new AssetFileJob(jobType!, Path.GetFullPath(asset.FullPath), outputPath));
                continue;
            }

            if (asset.Type == AssetType.Audio)
            {
                processedAudioCount++;

                var destination = FileSystemService.GetOutputPath(outputRoot, asset.RelativePath);

                if (IsNonEmptyFile(destination))
                    results.Add(new ConversionResult(
                        asset.RelativePath,
                        [destination],
                        ConversionStatus.Skipped,
                        "Audio file already exists; copy skipped."));
                else
                {
                    await fileSystem.CopyAsync(
                        asset.FullPath,
                        destination,
                        cancellationToken);

                    copiedAudioCount++;

                    results.Add(new ConversionResult(
                        asset.RelativePath,
                        [destination],
                        ConversionStatus.Copied));
                }

                now = DateTime.UtcNow;
                if (now - lastCheckProgress >= TimeSpan.FromSeconds(5))
                {
                    ConsoleReporter.Progress(
                        "CHECK",
                        $"Checking {processedAssetCount:N0}/{assets.Count:N0} files; " +
                        $"audio {processedAudioCount:N0}/{audioAssetCount:N0} processed, " +
                        $"{copiedAudioCount:N0} copied");

                    lastCheckProgress = now;
                }

                continue;
            }

            results.Add(new ConversionResult(
                asset.RelativePath,
                [],
                ConversionStatus.Skipped,
                "Not an output asset in asset-only mode."));
        }

        ConsoleReporter.Progress(
            "CHECK",
            $"Check complete: {processedAssetCount:N0}/{assets.Count:N0} files; " +
            $"audio {processedAudioCount:N0}/{audioAssetCount:N0} processed, " +
            $"{copiedAudioCount:N0} copied");

        ConsoleReporter.Complete();
        ConsoleReporter.Event("QUEUE", $"Prepared {jobs.Count} conversion job(s) from {assets.Count} files.");
        ConsoleReporter.Event(
            "BLENDER",
            jobs.Count == 0
                ? "No Blender conversion jobs queued."
                : $"Starting {jobs.Count} conversion job(s)");

        var orderedJobs = jobs
            .OrderBy(job => job.Type switch
            {
                "texture" => 0,
                "model" => 1,
                "map" => 2,
                _ => 3
            })
            .ToArray();

        results.AddRange(await blender.RunAsync(orderedJobs, cancellationToken));

        foreach (var result in results)
        {
            if (result.Status is ConversionStatus.Skipped or ConversionStatus.Copied)
                continue;
            var relativeInputPath = Path.IsPathRooted(result.InputPath)
                ? Path.GetRelativePath(inputRoot, result.InputPath)
                : result.InputPath;
            var details = result.Message
                ?? string.Join(
                    ", ",
                    result.OutputPaths.Select(path => Path.GetRelativePath(outputRoot, path)));
            ConsoleReporter.Progress(
                $"RESULT-{result.Status.ToString().ToUpperInvariant()}",
                $"{relativeInputPath}: {details}");
        }

        ConsoleReporter.Complete();

        await File.WriteAllTextAsync(
            reportPath,
            JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true }),
            cancellationToken);

        ConsoleReporter.Event("REPORT", reportPath);

        Console.WriteLine();
        Console.WriteLine($"Scanned: {assets.Count}");
        Console.WriteLine($"Results: {results.Count}");
        Console.WriteLine($"Converted: {results.Count(r => r.Status == ConversionStatus.Converted)}");
        Console.WriteLine($"Copied (audio): {results.Count(r => r.Status == ConversionStatus.Copied)}");
        Console.WriteLine($"Skipped: {results.Count(r => r.Status == ConversionStatus.Skipped)}");
        Console.WriteLine($"Failed: {results.Count(r => r.Status == ConversionStatus.Failed)}");
        Console.WriteLine($"Report: {reportPath}");
    }

    private HashSet<string> BuildExpectedOutputs(
        IReadOnlyList<AssetFile> assets,
        string outputRoot)
    {
        var expected = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void Add(string relativePath)
            => expected.Add(FileSystemService.GetOutputPath(outputRoot, relativePath));

        foreach (var asset in assets)
            switch (asset.Type)
            {
                case AssetType.Texture when options.ConvertTextures:
                    Add(Path.ChangeExtension(asset.RelativePath, ".png"));
                    break;
                case AssetType.Model when options.ConvertModels:
                    Add(Path.ChangeExtension(asset.RelativePath, ".glb"));
                    break;
                case AssetType.Map when options.ConvertMaps:
                    Add(Path.ChangeExtension(asset.RelativePath, ".blend"));
                    break;
                case AssetType.Audio:
                    Add(asset.RelativePath);
                    break;
            }

        return expected;
    }

    private static void CleanOutputDirectory(
        string outputRoot,
        string reportPath,
        HashSet<string> expectedOutputs)
    {
        foreach (var file in Directory.EnumerateFiles(outputRoot, "*", SearchOption.AllDirectories).ToArray())
        {
            var fullPath = Path.GetFullPath(file);
            if (string.Equals(fullPath, reportPath, StringComparison.OrdinalIgnoreCase))
                continue;
            if (!expectedOutputs.Contains(fullPath))
            {
                ConsoleReporter.Progress("DELETE", Path.GetRelativePath(outputRoot, fullPath));
                File.Delete(fullPath);
            }
        }
        ConsoleReporter.Complete();
        foreach (var directory in Directory
             .EnumerateDirectories(outputRoot, "*", SearchOption.AllDirectories)
             .OrderByDescending(path => path.Length)
             .ToArray())
        {
            if (Directory.EnumerateFileSystemEntries(directory).Any())
                continue;
            ConsoleReporter.Progress("RMDIR", Path.GetRelativePath(outputRoot, directory));
            Directory.Delete(directory);
        }

        ConsoleReporter.Complete();
    }

    private bool TryGetConversion(
        AssetFile asset,
        out string? jobType,
        out string? outputExtension)
    {
        jobType = null;
        outputExtension = null;

        switch (asset.Type)
        {
            case AssetType.Texture when options.ConvertTextures:
                jobType = "texture";
                outputExtension = ".png";
                return true;
            case AssetType.Model when options.ConvertModels:
                jobType = "model";
                outputExtension = ".glb";
                return true;
            case AssetType.Map when options.ConvertMaps:
                jobType = "map";
                outputExtension = ".blend";
                return true;
            default:
                return false;
        }
    }

    private static bool IsNonEmptyFile(string path)
    {
        try
        {
            return File.Exists(path) && new FileInfo(path).Length > 0;
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static bool IsValidPng(string path)
    {
        if (!IsNonEmptyFile(path))
            return false;

        try
        {
            using var stream = File.OpenRead(path);
            var signature = new byte[PngSignature.Length];
            return stream.Read(signature, 0, signature.Length) == signature.Length
                && signature.SequenceEqual(PngSignature);
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }


    private static bool IsValidGlb(string path)
    {
        if (!IsNonEmptyFile(path))
            return false;

        try
        {
            var data = File.ReadAllBytes(path);
            if (data.Length < 28 ||
                data[0] != (byte)'g' ||
                data[1] != (byte)'l' ||
                data[2] != (byte)'T' ||
                data[3] != (byte)'F')
                return false;
            if (BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(4, 4)) != 2)
                return false;
            if (BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(8, 4)) != (uint)data.Length)
                return false;

            var jsonLength = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(12, 4));
            var jsonType = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(16, 4));
            if (jsonType != 0x4E4F534A ||
                jsonLength == 0 ||
                jsonLength > (uint)(data.Length - 28))
                return false;

            using var document = JsonDocument.Parse(data.AsMemory(20, checked((int)jsonLength)));

            var root = document.RootElement;
            if (!root.TryGetProperty("meshes", out var meshes) ||
                meshes.ValueKind != JsonValueKind.Array ||
                !meshes.EnumerateArray().Any(mesh =>
                    mesh.ValueKind == JsonValueKind.Object &&
                    mesh.TryGetProperty("primitives", out var primitives) &&
                    primitives.ValueKind == JsonValueKind.Array &&
                    primitives.GetArrayLength() > 0))
                return false;

            var binHeaderOffset = checked(20 + (int)jsonLength);
            if (binHeaderOffset + 8 > data.Length)
                return false;

            var binLength = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(binHeaderOffset, 4));
            var binType = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(binHeaderOffset + 4, 4));
            if (binType != 0x004E4942 ||
                binLength == 0 ||
                binLength > (uint)(data.Length - binHeaderOffset - 8))
                return false;
            if (root.TryGetProperty("images", out var images))
            {
                if (images.ValueKind != JsonValueKind.Array)
                    return false;

                foreach (var image in images.EnumerateArray())
                {
                    var hasBufferView =
                        image.ValueKind == JsonValueKind.Object &&
                        image.TryGetProperty("bufferView", out var bufferView) &&
                        bufferView.ValueKind == JsonValueKind.Number;
                    var hasDataUri =
                        image.ValueKind == JsonValueKind.Object &&
                        image.TryGetProperty("uri", out var imageUri) &&
                        imageUri.ValueKind == JsonValueKind.String &&
                        imageUri.GetString()!.StartsWith(
                            "data:",
                            StringComparison.OrdinalIgnoreCase);

                    if (!hasBufferView && !hasDataUri)
                        return false;
                }
            }
            if (root.TryGetProperty("buffers", out var buffers))
            {
                if (buffers.ValueKind != JsonValueKind.Array)
                    return false;
                foreach (var buffer in buffers.EnumerateArray())
                {
                    if (buffer.ValueKind != JsonValueKind.Object || !buffer.TryGetProperty("uri", out var uri))
                        continue;
                    if (uri.ValueKind == JsonValueKind.String &&
                        !uri.GetString()!.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                        return false;
                }
            }

            return true;
        }
        catch (Exception ex) when (
            ex is IOException or UnauthorizedAccessException or
            JsonException or ArgumentException or OverflowException)
        {
            return false;
        }
    }
}
