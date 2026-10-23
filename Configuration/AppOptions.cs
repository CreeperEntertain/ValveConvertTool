namespace ValveConvertTool.Configuration;

internal sealed class AppOptions
{
    public required string InputDirectory { get; init; }
    public required string OutputDirectory { get; init; }
    public required string BlenderExecutable { get; init; }

    public bool CopyBspFiles { get; init; } = false;
    public bool CopyAudioFiles { get; init; } = true;
    public bool PreserveRelativePaths { get; init; } = true;
    public bool DiscardUnsupportedFiles { get; init; } = true;
}
