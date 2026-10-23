namespace ValveConvertTool.Core;

internal sealed record ConversionResult(
    string InputPath,
    IReadOnlyList<string> OutputPaths,
    ConversionStatus Status,
    string? Message = null);

public enum ConversionStatus
{
    Converted,
    Copied,
    Skipped,
    Failed
}
