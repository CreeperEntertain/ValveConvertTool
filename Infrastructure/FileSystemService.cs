namespace ValveConvertTool.Infrastructure;

internal sealed class FileSystemService
{
    public static string NormalizeDirectory(string path)
        => Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

    public static void ValidateDirectories(
    string inputDirectory,
    string outputDirectory)
    {
        var input = NormalizeDirectory(inputDirectory);
        var output = NormalizeDirectory(outputDirectory);
        if (string.Equals(input, output, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Input and output directories must be different.");
        if (output.StartsWith(input + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Output cannot be inside the input directory.");
        if (input.StartsWith(output + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                "Input cannot be inside the output directory; output cleanup would be unsafe.");
    }

    public static string GetOutputPath(
    string outputRoot,
    string relativePath)
    {
        var root = Path.GetFullPath(outputRoot);
        var destination = Path.GetFullPath(Path.Combine(root, relativePath));
        if (!destination.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Invalid relative path: {relativePath}");

        return destination;
    }

    public async Task CopyAsync(
    string sourcePath,
    string destinationPath,
    CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);

        await using var source = File.OpenRead(sourcePath);
        await using var destination = File.Create(destinationPath);

        await source.CopyToAsync(destination, cancellationToken);
    }
}
