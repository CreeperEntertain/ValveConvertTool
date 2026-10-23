namespace ValveConvertTool.Configuration;

internal sealed class AppOptions
{
    public required string InputDirectory { get; init; }
    public required string OutputDirectory { get; init; }
    public required string BlenderExecutable { get; init; }

    public bool ConvertTextures { get; init; } = true;
    public bool ConvertMaterials { get; init; } = true;
    public bool ConvertModels { get; init; } = true;
    public bool ConvertMaps { get; init; } = true;
}
