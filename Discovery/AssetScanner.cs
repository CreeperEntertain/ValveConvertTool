using ValveConvertTool.Core;
using ValveConvertTool.Infrastructure;

namespace ValveConvertTool.Discovery;

internal sealed record AssetFile(
    string FullPath,
    string RelativePath,
    AssetType Type);

internal sealed class AssetScanner
{
    private static readonly HashSet<string> ModelCompanions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".vvd",
        ".vtx",
        ".phy",
        ".ani"
    };
    private static readonly HashSet<string> PreservedSourceFormats = new(StringComparer.OrdinalIgnoreCase)
    {
        ".qc",
        ".qci",
        ".smd",
        ".vta",
        ".vmf",
        ".vmx",
        ".dmx",
        ".txt",
        ".cfg",
        ".res",
        ".nut",
        ".kv",
        ".kv2",
        ".kv3",
        ".vdf",
        ".fgd",
        ".rad",
        ".nav",
        ".ain",
        ".pop",
        ".manifest",
        ".json",
        ".xml",
        ".csv",
        ".lua",
        ".ini"
    };

    public IReadOnlyList<AssetFile> Scan(string inputDirectory)
    {
        var root = Path.GetFullPath(inputDirectory);
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException(root);

        var assets = new List<AssetFile>();
        var lastScanProgress = DateTime.MinValue;

        foreach (var path in Directory.EnumerateFiles(
            root,
            "*",
            SearchOption.AllDirectories))
        {
            var relativePath = Path.GetRelativePath(root, path);

            assets.Add(new AssetFile(
                path,
                relativePath,
                Classify(Path.GetExtension(path))));

            var now = DateTime.UtcNow;
            if (assets.Count == 1 || now - lastScanProgress >= TimeSpan.FromSeconds(5))
            {
                ConsoleReporter.Progress(
                    "SCAN",
                    $"{assets.Count:N0} files discovered");

                lastScanProgress = now;
            }
        }

        ConsoleReporter.Progress(
            "SCAN",
            $"Scan complete: {assets.Count:N0} files discovered");

        ConsoleReporter.Complete();

        return assets
            .OrderBy(
                asset => asset.RelativePath,
                StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    public static AssetType Classify(string extension)
    {
        extension = extension.ToLowerInvariant();
        if (ModelCompanions.Contains(extension))
            return AssetType.ModelCompanion;
        if (PreservedSourceFormats.Contains(extension))
            return AssetType.SourceDocument;

        return extension switch
        {
            ".vtf" => AssetType.Texture,
            ".vmt" => AssetType.Material,
            ".mdl" => AssetType.Model,
            ".bsp" => AssetType.Map,
            ".wav" or ".mp3" or ".ogg" => AssetType.Audio,
            _ => AssetType.Other
        };
    }
}
