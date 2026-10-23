namespace ValveConvertTool.Infrastructure;

internal static class ConsoleReporter
{
    private static readonly object Gate = new();
    private static string? _activeType;
    private static int _lastWidth;

    public static void Progress(string type, string message)
    {
        lock (Gate)
        {
            if (_activeType != type)
            {
                if (_activeType is not null)
                    Console.WriteLine();
                _activeType = type;
                _lastWidth = 0;
            }
            var text = $"[{type}] {message}";
            Console.Write("\r" + text.PadRight(_lastWidth));
            _lastWidth = Math.Max(_lastWidth, text.Length);
        }
    }

    public static void Complete()
    {
        lock (Gate)
        {
            if (_activeType is null)
                return;
            Console.WriteLine();
            _activeType = null;
            _lastWidth = 0;
        }
    }

    public static void Event(string type, string message)
    {
        Complete();
        Console.WriteLine($"[{type}] {message}");
    }
}
