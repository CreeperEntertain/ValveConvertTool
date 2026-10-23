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

        foreach (var asset in assets)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (TryGetConversion(asset, out var jobType, out var outputExtension))
            {
                var outputRelative = Path.ChangeExtension(asset.RelativePath, outputExtension!);
                var outputPath = FileSystemService.GetOutputPath(outputRoot, outputRelative);
                var mtlPath = Path.ChangeExtension(outputPath, ".mtl");

                if (asset.Type == AssetType.Texture && IsValidPng(outputPath))
                {
                    ConsoleReporter.Progress("SKIP", $"{asset.RelativePath} — valid PNG already exists");
                    results.Add(new ConversionResult(
                        asset.RelativePath, [outputPath], ConversionStatus.Skipped,
                        "Valid output already exists; conversion skipped."));
                    continue;
                }
                if (asset.Type == AssetType.Model && IsValidObjWithMtl(outputPath, mtlPath))
                {
                    ConsoleReporter.Progress("SKIP", $"{asset.RelativePath} — valid OBJ and MTL already exist");
                    results.Add(new ConversionResult(
                        asset.RelativePath, [outputPath, mtlPath], ConversionStatus.Skipped,
                        "OBJ geometry and its MTL already exist; conversion skipped."));
                    continue;
                }
                if (asset.Type == AssetType.Map && IsNonEmptyFile(outputPath))
                {
                    ConsoleReporter.Progress("SKIP", $"{asset.RelativePath} — BLEND output already exists");
                    results.Add(new ConversionResult(
                        asset.RelativePath, [outputPath], ConversionStatus.Skipped,
                        "BLEND output for this explicit BSP already exists; conversion skipped."));
                    continue;
                }

                ConsoleReporter.Progress("QUEUE", $"{asset.Type}: {asset.RelativePath}");

                jobs.Add(new AssetFileJob(jobType!, Path.GetFullPath(asset.FullPath), outputPath));
                continue;
            }

            if (asset.Type == AssetType.Audio)
            {
                var destination = FileSystemService.GetOutputPath(outputRoot, asset.RelativePath);
                if (IsNonEmptyFile(destination))
                {
                    ConsoleReporter.Progress("SKIP", $"{asset.RelativePath} — audio already copied");
                    results.Add(new ConversionResult(
                        asset.RelativePath, [destination], ConversionStatus.Skipped,
                        "Audio file already exists; copy skipped."));
                    continue;
                }

                ConsoleReporter.Progress("COPY", $"Copying {asset.RelativePath}");
                await fileSystem.CopyAsync(asset.FullPath, destination, cancellationToken);
                ConsoleReporter.Progress("COPY", $"Copied {asset.RelativePath}");
                results.Add(new ConversionResult(asset.RelativePath, [destination], ConversionStatus.Copied));
                continue;
            }

            ConsoleReporter.Progress("SKIP", $"{asset.RelativePath} — not an output asset in asset-only mode");

            results.Add(new ConversionResult(
                asset.RelativePath,
                [],
                ConversionStatus.Skipped,
                "Not an output asset in asset-only mode."));
        }

        ConsoleReporter.Complete();
        ConsoleReporter.Event(
            "BLENDER",
            jobs.Count == 0
                ? "No Blender conversion jobs queued."
                : $"Starting {jobs.Count} conversion job(s)");

        results.AddRange(await blender.RunAsync(jobs, cancellationToken));

        foreach (var result in results)
        {
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
                if (line.StartsWith("v ", StringComparison.Ordinal))
                    hasVertex = true;
                else if (line.StartsWith("f ", StringComparison.Ordinal))
                    hasFace = true;
                if (hasVertex && hasFace)
                    return true;
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
}
