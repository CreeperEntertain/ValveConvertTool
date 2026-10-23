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

        foreach (var asset in assets)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var outputRelative = asset.RelativePath;

            switch (asset.Type)
            {
                case AssetType.Texture when options.ConvertTextures:
                    outputRelative = Path.ChangeExtension(asset.RelativePath, ".png");
                    jobs.Add(CreateJob("texture", asset, outputRelative));
                    break;
                case AssetType.Model when options.ConvertModels:
                    outputRelative = Path.ChangeExtension(asset.RelativePath, ".obj");
                    jobs.Add(CreateJob("model", asset, outputRelative));
                    break;
                case AssetType.Map when options.ConvertMaps:
                    outputRelative = Path.ChangeExtension(asset.RelativePath, ".blend");
                    jobs.Add(CreateJob("map", asset, outputRelative));
                    break;
                case AssetType.Audio when options.CopyAudio:
                case AssetType.SourceDocument:
                    {
                        var destination = FileSystemService.GetOutputPath(options.OutputDirectory, outputRelative);
                        await fileSystem.CopyAsync(asset.FullPath, destination, cancellationToken);
                        results.Add(new ConversionResult(asset.RelativePath, [destination], ConversionStatus.Copied));
                        break;
                    }
                case AssetType.ModelCompanion:
                    // MDL conversion, not outputs
                    break;
            }
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

    private AssetFileJob CreateJob(string type, AssetFile asset, string relativeOutput)
        => new AssetFileJob(
            type,
            Path.GetFullPath(asset.FullPath),
            FileSystemService.GetOutputPath(options.OutputDirectory, relativeOutput));
}
