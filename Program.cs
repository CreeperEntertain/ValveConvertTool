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
                BlenderExecutable = Required(values, "blender"),
                SteamAppId = int.TryParse(
                    values.GetValueOrDefault("steam-app-id", "400"),
                    out var appId) ? appId : 400
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
            Console.Error.WriteLine($"Error: {ex.Message}");
            Console.ReadKey();
            return 1;
        }
    }

    private static Dictionary<string, string> ParseArguments(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < args.Length; i++)
        {
            var key = args[i].TrimStart('-');
            if (key.Equals("help", StringComparison.OrdinalIgnoreCase))
            {
                values["help"] = "";
                continue;
            }
            if (!key.Equals(args[i], StringComparison.Ordinal) || args[i].StartsWith('-'))
            {
                if (i + 1 >= args.Length || args[i + 1].StartsWith("--"))
                    throw new ArgumentException($"Missing value for argument: {args[i]}");
                values[key] = args[++i];
            }
            else
                throw new ArgumentException($"Unexpected argument: {args[i]}");
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
            ValveConvertTool — Portal 1 asset converter

            Usage:
              ValveConvertTool --input <folder> --output <folder>
                               --blender <blender.exe>
                               [--steam-app-id 400]

            Examples:
              ValveConvertTool --input "D:\PortalExtracted"
                               --output "D:\PortalConverted"
                               --blender "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
                               --steam-app-id 400
            """);
    }
}
