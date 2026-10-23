using System.Diagnostics;
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

        try
        {
            await File.WriteAllTextAsync(
                manifestPath,
                JsonSerializer.Serialize(new
                {
                    input_root = Path.GetFullPath(options.InputDirectory),
                    steam_app_id = options.SteamAppId,
                    jobs
                }),
                cancellationToken);

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

            var stdoutTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
            var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);

            await process.WaitForExitAsync(cancellationToken);

            var stdout = await stdoutTask;
            var stderr = await stderrTask;
            if (process.ExitCode != 0)
                throw new InvalidOperationException(
                    $"Blender failed with exit code {process.ExitCode}.\n" +
                    $"STDOUT:\n{stdout}\nSTDERR:\n{stderr}");

            var resultPath = Path.ChangeExtension(manifestPath, ".results.json");
            if (!File.Exists(resultPath))
                throw new InvalidOperationException($"Blender completed without writing results.\n{stdout}\n{stderr}");

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
            File.Delete(Path.ChangeExtension(manifestPath, ".results.json"));
        }
    }
}

internal sealed record AssetFileJob(
    string Type,
    string InputPath,
    string OutputPath);
