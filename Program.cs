using ValveConvertTool.Configuration;
using ValveConvertTool.Discovery;
using ValveConvertTool.Infrastructure;
using ValveConvertTool.Pipeline;

namespace ValveConvertTool;

internal class Program
{
    static async Task<int> Main(string[] args)
    {
        try
        {
            var values = ParseArguments(args);
            if (values.ContainsKey("help"))
            {
                PrintHelp();
                return 0;
            }

            var options = new AppOptions
            {
                InputDirectory = Required(values, "input"),
                OutputDirectory = Required(values, "output"),
                BlenderExecutable = Required(values, "blender")
            };

            FileSystemService.ValidateDirectories(options.InputDirectory, options.OutputDirectory);

            var scanner = new AssetScanner();
            var fileSystem = new FileSystemService();
            var blender = new BlenderRunner(options);

            var pipeline = new ConversionPipeline(options, scanner, fileSystem, blender);

            await pipeline.RunAsync();
            return 0;
        }
        catch (Exception ex)
        {
            ConsoleReporter.Complete();
            Console.Error.WriteLine($"Error: {ex.Message}");
            Console.ReadKey();
            return 1;
        }
    }

    private static Dictionary<string, string> ParseArguments(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var allowed = new HashSet<string>(["input", "output", "blender"], StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < args.Length; i++)
        {
            var argument = args[i];
            if (argument is "--help" or "-h")
            {
                values["help"] = "";
                continue;
            }
            if (!argument.StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException($"Unexpected argument: {argument}");

            var key = argument[2..];
            if (!allowed.Contains(key))
                throw new ArgumentException($"Unknown argument: {argument}");
            if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException($"Missing value for argument: {argument}");

            values[key] = args[++i];
        }

        return values;
    }

    private static string Required(Dictionary<string, string> values, string key)
    {
        if (!values.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"Missing required argument: --{key}");
        return value;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
            ValveConvertTool — Source Engine asset converter

            Usage:
              ValveConvertTool --input <folder>
                               --output <folder>
                               --blender <blender.exe>

            Example:
              ValveConvertTool --input "D:\SourceAssets"
                               --output "D:\ConvertedAssets"
                               --blender "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
            """);
    }
}
