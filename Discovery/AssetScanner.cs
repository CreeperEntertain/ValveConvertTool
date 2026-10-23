using ValveConvertTool.Core;

namespace ValveConvertTool.Discovery;

internal sealed record AssetFile(
    string FullPath,
    string RelativePath,
    AssetType Type);

internal sealed class AssetScanner
{
    private static readonly HashSet<string> ModelCompanions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".vvd", ".vtx", ".phy", ".ani"
    };
    private static readonly HashSet<string> PreservedSourceFormats = new(StringComparer.OrdinalIgnoreCase)
    {
        ".vmt", ".qc", ".smd", ".vta", ".vmf", ".vmx"
    };

    public IReadOnlyList<AssetFile> Scan(string inputDirectory)
    {
        var root = Path.GetFullPath(inputDirectory);
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException(root);

        return Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
           .Select(path => new AssetFile(
               path,
               Path.GetRelativePath(root, path),
               Classify(Path.GetExtension(path))))
           .Where(asset => asset.Type != AssetType.Unsupported)
           .OrderBy(asset => asset.RelativePath,
               StringComparer.OrdinalIgnoreCase)
           .ToArray();
    }

    public static AssetType Classify(string extension)
    {
        if (ModelCompanions.Contains(extension))
            return AssetType.ModelCompanion;
        if (PreservedSourceFormats.Contains(extension))
            return AssetType.SourceDocument;

        return extension.ToLowerInvariant() switch
        {
            ".vtf" => AssetType.Texture,
            ".mdl" => AssetType.Model,
            ".bsp" => AssetType.Map,
            ".wav" or ".mp3" => AssetType.Audio,
            _ => AssetType.Unsupported
        };
    }
}
