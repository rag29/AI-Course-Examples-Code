using Microsoft.ML;
using System.Text.RegularExpressions;

public class TextData {
    public string Text { get; set; }
}

public class NgramPrediction {
    public string Input { get; set; }
    public string Label { get; set; }
}

public class Program {
    static Dictionary<string, Dictionary<string, int>> ngramModel = new Dictionary<string, Dictionary<string, int>>();

    static IEnumerable<TextData> LoadData(MLContext mlContext, string inputPath) {
        string text = File.ReadAllText(inputPath);
        return new List<TextData> { new TextData { Text = text } };
    }

    static void CreateNgramModel(IEnumerable<TextData> data, int n) {
        foreach(var text in data) {
            var words = Regex.Replace(text.Text, @"[^\w\s]", "").ToLower().Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            for(int i = 0; i <= words.Length - n; i++) {
                var ngramKey = string.Join(" ", words.Skip(i).Take(n-1));
                var nextWord = words[i + n - 1];
                if(!ngramModel.ContainsKey(ngramKey)) {
                    ngramModel[ngramKey] = new Dictionary<string, int>();
                }
                if(!ngramModel[ngramKey].ContainsKey(nextWord)) {
                    ngramModel[ngramKey][nextWord] = 0;
                }
                ngramModel[ngramKey][nextWord]++;
            }
        }
    }

    static string GetRandomWord(Dictionary<string, int> nextWords) {
        var total = nextWords.Values.Sum();
        var randomValue = new Random().Next(0, total);
        foreach(var word in nextWords) {
            randomValue -= word.Value;
            if(randomValue < 0) {
                return word.Key;
            }
        }
        return string.Empty;
    }

    static string GenerateText(string seed, int length) {
        var result = seed;
        var words = seed.Split(' ');

        for(int i = 0; i < length; i++) {
            var ngramKey = string.Join(" ", words.Skip(Math.Max(0, words.Length - 1)));

            if(ngramModel.ContainsKey(ngramKey)) {
                var nextWords = ngramModel[ngramKey];
                var nextWord = GetRandomWord(nextWords);
                result += " " + nextWord;

                Array.Resize(ref words, words.Length + 1);
                words[words.Length - 1] = nextWord;
            } else {
                break;
            }
        }
        return result;
    }

    public static void Main(string[] args) {
        var mlContext = new MLContext();
        var data = LoadData(mlContext, "input.txt");
        CreateNgramModel(data, n: 2);
        Console.WriteLine("Enter a starting sentence:");
        string seed = Console.ReadLine();
        Console.WriteLine("How many words would you like to generate?");
        int length = int.Parse(Console.ReadLine());
        string generatedText = GenerateText(seed, length);
        Console.WriteLine("\nGenerated Shakespearean-style text:");
        Console.WriteLine(generatedText);
    }
}
