namespace ValveConvertTool.Core;

internal interface IAssetConverter
{
    bool CanConvert(string extension);

    Task<ConversionStatus> ConvertAsync(
        string inputPath,
        string outputDirectory,
        CancellationToken cancellationToken = default);
}
