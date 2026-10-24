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
        PreserveReferencedMtlTextures(outputRoot, expectedOutputs);
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
                var mtlPath = Path.ChangeExtension(outputPath, ".mtl");

                if (asset.Type == AssetType.Texture && IsValidPng(outputPath))
                {
                    results.Add(new ConversionResult(
                        asset.RelativePath, [outputPath], ConversionStatus.Skipped,
                        "Valid output already exists; conversion skipped."));
                    continue;
                }
                if (asset.Type == AssetType.Model && IsValidObjWithMtl(outputPath, mtlPath))
                {
                    results.Add(new ConversionResult(
                        asset.RelativePath, [outputPath, mtlPath], ConversionStatus.Skipped,
                        "OBJ geometry and its MTL already exist; conversion skipped."));
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

    private static void PreserveReferencedMtlTextures(
        string outputRoot,
        HashSet<string> expectedOutputs)
    {
        if (!Directory.Exists(outputRoot))
            return;

        var outputPrefix = Path.GetFullPath(outputRoot)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;

        foreach (var mtlPath in Directory.EnumerateFiles(outputRoot, "*.mtl", SearchOption.AllDirectories))
        {
            var fullMtlPath = Path.GetFullPath(mtlPath);
            if (!expectedOutputs.Contains(fullMtlPath))
                continue;

            try
            {
                foreach (var line in File.ReadLines(fullMtlPath))
                {
                    var texturePath = ResolveMtlTexturePath(fullMtlPath, line);
                    if (texturePath is null || !File.Exists(texturePath))
                        continue;

                    var fullTexturePath = Path.GetFullPath(texturePath);
                    if (fullTexturePath.StartsWith(outputPrefix, StringComparison.OrdinalIgnoreCase))
                        expectedOutputs.Add(fullTexturePath);
                }
            }
            catch (IOException)
            {
                // Damaged old MTLs shouldn't not stop the conversion pipeline
            }
            catch (UnauthorizedAccessException)
            {
                // Unreadable old MTLs must not shouldn't the conversion pipeline
            }
            // I feel like Bart Simpson at the chalkboard writing these
        }
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
                    var objRelative = Path.ChangeExtension(asset.RelativePath, ".obj");
                    Add(objRelative);
                    Add(Path.ChangeExtension(objRelative, ".mtl"));
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
                outputExtension = ".obj";
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

    private static bool IsValidObjWithMtl(string objPath, string mtlPath)
    {
        if (!IsNonEmptyFile(objPath) || !IsNonEmptyFile(mtlPath))
            return false;

        try
        {
            var hasVertex = false;
            var hasFace = false;

            foreach (var line in File.ReadLines(objPath))
            {
                if (line.StartsWith("v ", StringComparison.Ordinal))
                    hasVertex = true;
                else if (line.StartsWith("f ", StringComparison.Ordinal))
                    hasFace = true;
            }

            if (!hasVertex || !hasFace)
                return false;

            foreach (var line in File.ReadLines(mtlPath))
            {
                var fields = line.Trim().Split(
                    (char[]?)null,
                    StringSplitOptions.RemoveEmptyEntries);

                if (fields.Length > 1 && fields[0].Equals("newmtl", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }

        return false;
    }

    private static string? ResolveMtlTexturePath(string mtlPath, string line)
    {
        var fields = line.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (fields.Length < 2 ||
            !fields[0].StartsWith("map_", StringComparison.OrdinalIgnoreCase))
            return null;

        var index = 1;

        while (index < fields.Length && fields[index].StartsWith("-", StringComparison.Ordinal))
            switch (fields[index])
            {
                case "-o":
                case "-s":
                    index += 4;
                    break;

                case "-bm":
                case "-boost":
                case "-texres":
                case "-clamp":
                case "-blendu":
                case "-blendv":
                case "-imfchan":
                case "-type":
                    index += 2;
                    break;

                default:
                    return null;
            }
        if (index >= fields.Length)
            return null;

        var reference = string.Join(" ", fields.Skip(index)).Trim().Trim('"');
        if (string.IsNullOrWhiteSpace(reference))
            return null;

        reference = reference.Replace('/', Path.DirectorySeparatorChar);

        return Path.IsPathRooted(reference)
            ? Path.GetFullPath(reference)
            : Path.GetFullPath(Path.Combine(Path.GetDirectoryName(mtlPath)!, reference));
    }
}
