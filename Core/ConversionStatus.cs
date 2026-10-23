namespace ValveConvertTool.Core;

public enum ConversionStatus
{
    Converted,
    Copied,
    Skipped,
    Failed
}

internal sealed record ConversionResult(
    string InputPath,
    IReadOnlyList<string> OutputPaths,
    ConversionStatus Status,
    string? Message = null);
