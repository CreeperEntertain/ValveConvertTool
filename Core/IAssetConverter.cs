namespace ValveConvertTool.Core;

internal interface IAssetConverter
{
    bool CanConvert(string extension);

    Task<ConversionResult> ConvertAsync(
        string inputPath,
        string outputDirectory,
        CancellationToken cancellationToken);
}
