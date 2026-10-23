using System.Diagnostics;
using System.Text;
using System.Text.Json;
using ValveConvertTool.Configuration;
using ValveConvertTool.Core;

namespace ValveConvertTool.Infrastructure;

internal sealed class BlenderRunner(AppOptions options)
{
    public async Task<IReadOnlyList<ConversionResult>> RunAsync(
        IReadOnlyList<AssetFileJob> jobs,
        CancellationToken cancellationToken = default)
    {
        if (jobs.Count == 0)
            return [];

        var manifestPath = Path.Combine(Path.GetTempPath(), $"valveconvert-{Guid.NewGuid():N}.json");
        var resultPath = Path.ChangeExtension(manifestPath, ".results.json");

        try
        {
            var manifest = new
            {
                input_root = Path.GetFullPath(options.InputDirectory),
                jobs
            };

            await File.WriteAllTextAsync(manifestPath, JsonSerializer.Serialize(manifest), cancellationToken);
            var workerPath = Path.Combine(AppContext.BaseDirectory, "Scripts", "blender_worker.py");
            if (!File.Exists(workerPath))
                throw new FileNotFoundException("Blender worker script not found.", workerPath);

            var startInfo = new ProcessStartInfo
            {
                FileName = options.BlenderExecutable,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            startInfo.ArgumentList.Add("--background");
            startInfo.ArgumentList.Add("--python");
            startInfo.ArgumentList.Add(workerPath);
            startInfo.ArgumentList.Add("--");
            startInfo.ArgumentList.Add(manifestPath);

            using var process = new Process { StartInfo = startInfo };
            process.Start();

            var stdoutTask = ReadOutputAsync(process.StandardOutput, "BLENDER", cancellationToken);
            var stderrTask = ReadOutputAsync(process.StandardError, "BLENDER-ERROR", cancellationToken);

            await process.WaitForExitAsync(cancellationToken);

            var stdout = await stdoutTask;
            var stderr = await stderrTask;

            ConsoleReporter.Complete();
            ConsoleReporter.Event("BLENDER", $"Process exited with code {process.ExitCode}");

            if (process.ExitCode != 0)
                throw new InvalidOperationException(
                    $"Blender failed with exit code {process.ExitCode}.\n" +
                    $"STDOUT:\n{stdout}\nSTDERR:\n{stderr}");
            if (!File.Exists(resultPath))
                throw new InvalidOperationException(
                    "Blender completed without writing results.\n" +
                    $"STDOUT:\n{stdout}\nSTDERR:\n{stderr}");

            return JsonSerializer.Deserialize<List<ConversionResult>>(
                await File.ReadAllTextAsync(resultPath, cancellationToken),
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                }) ?? [];
        }
        finally
        {
            File.Delete(manifestPath);
            File.Delete(resultPath);
        }
    }

    private static async Task<string> ReadOutputAsync(
        StreamReader reader,
        string type,
        CancellationToken cancellationToken)
    {
        var output = new StringBuilder();

        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            output.AppendLine(line);

            if (type == "BLENDER-ERROR")
            {
                ConsoleReporter.Event(type, line);
                continue;
            }

            var progressType = type;

            if (line.StartsWith("[", StringComparison.Ordinal))
            {
                var end = line.IndexOf(']');
                if (end > 1)
                    progressType = "CONVERT-" + line[1..end].ToUpperInvariant();
            }

            ConsoleReporter.Progress(progressType, line);
        }

        return output.ToString();
    }
}

internal sealed record AssetFileJob(
    string Type,
    string InputPath,
    string OutputPath);
