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
    public async Task RunAsync(CancellationToken cancellationToken = default)
    {
        FileSystemService.ValidateDirectories(options.InputDirectory, options.OutputDirectory);

        Directory.CreateDirectory(options.OutputDirectory);

        var assets = scanner.Scan(options.InputDirectory);
        var jobs = new List<AssetFileJob>();
        var results = new List<ConversionResult>();

        var reservedPaths = assets
            .Where(asset => !WillConvert(asset))
            .Select(asset => FileSystemService.GetOutputPath(
                options.OutputDirectory,
                asset.RelativePath))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (var asset in assets)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (TryGetConversion(asset, out var jobType, out var outputExtension))
            {
                var outputRelative = Path.ChangeExtension(asset.RelativePath, outputExtension);
                var outputPath = GetUniqueOutputPath(outputRelative, reservedPaths);
                jobs.Add(new AssetFileJob(jobType!, Path.GetFullPath(asset.FullPath), outputPath));
                continue;
            }

            var destination = FileSystemService.GetOutputPath(options.OutputDirectory, asset.RelativePath);
            await fileSystem.CopyAsync(asset.FullPath, destination, cancellationToken);

            results.Add(new ConversionResult(
                asset.RelativePath,
                [destination],
                ConversionStatus.Copied,
                asset.Type == AssetType.Other
                    ? "No dedicated converter for this extension; preserved unchanged."
                    : null));
        }

        results.AddRange(await blender.RunAsync(jobs, cancellationToken));
        var reportPath = Path.Combine(options.OutputDirectory, "conversion-report.json");
        await File.WriteAllTextAsync(
            reportPath,
            JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true }),
            cancellationToken);

        Console.WriteLine();
        Console.WriteLine($"Scanned: {assets.Count}");
        Console.WriteLine($"Results: {results.Count}");
        Console.WriteLine($"Converted: {results.Count(r => r.Status == ConversionStatus.Converted)}");
        Console.WriteLine($"Copied: {results.Count(r => r.Status == ConversionStatus.Copied)}");
        Console.WriteLine($"Failed: {results.Count(r => r.Status == ConversionStatus.Failed)}");
        Console.WriteLine($"Report: {reportPath}");
    }

    private bool WillConvert(AssetFile asset)
        => asset.Type switch
        {
            AssetType.Texture => options.ConvertTextures,
            AssetType.Material => options.ConvertMaterials,
            AssetType.Model => options.ConvertModels,
            AssetType.Map => options.ConvertMaps,
            _ => false
        };

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
            case AssetType.Material when options.ConvertMaterials:
                jobType = "material";
                outputExtension = ".blend";
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

    private string GetUniqueOutputPath(
        string relativePath,
        HashSet<string> reservedPaths)
    {
        var candidate = FileSystemService.GetOutputPath(options.OutputDirectory, relativePath);
        if (reservedPaths.Add(candidate))
            return candidate;

        var directory = Path.GetDirectoryName(relativePath);
        var stem = Path.GetFileNameWithoutExtension(relativePath);
        var extension = Path.GetExtension(relativePath);

        for (var index = 1; ; index++)
        {
            var fileName = $"{stem}.converted{index}{extension}";
            var alternateRelative = string.IsNullOrEmpty(directory) ? fileName : Path.Combine(directory, fileName);

            candidate = FileSystemService.GetOutputPath(options.OutputDirectory, alternateRelative);
            if (reservedPaths.Add(candidate))
                return candidate;
        }
    }
}
