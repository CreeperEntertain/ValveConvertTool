namespace ValveConvertTool.Configuration;

internal sealed class AppOptions
{
    public required string InputDirectory { get; init; }
    public required string OutputDirectory { get; init; }
    public required string BlenderExecutable { get; init; }

    public int SteamAppId { get; set; } = 400;

    public bool CopyAudio { get; set; } = true;
    public bool ConvertTextures { get; init; } = true;
    public bool ConvertModels { get; init; } = true;
    public bool ConvertMaps { get; init; } = true;
    public bool PreserveRelativePaths { get; init; } = true;
}
