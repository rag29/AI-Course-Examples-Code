using Microsoft.ML;
using Microsoft.ML.Data;

namespace StockPriceForecasting {
    class Program {
        public class StockData {
            [LoadColumn(0)]
            public string Date { get; set; }

            [LoadColumn(1)]
            public float Open { get; set; }

            [LoadColumn(2)]
            public float High { get; set; }

            [LoadColumn(3)]
            public float Low { get; set; }

            [LoadColumn(4)]
            public float Close { get; set; }
        }

        public class StockPrediction {
            [ColumnName("Score")]
            public float PredictedClose { get; set; }
        }

        static void Main(string[] args) {
            var mlContext = new MLContext();
            var dataView = mlContext.Data.LoadFromTextFile<StockData>("stock_data.csv", separatorChar: ',');
            var preview = dataView.Preview();
            foreach(var row in preview.RowView) {
                Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
            }
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(StockData.Open), nameof(StockData.High), nameof(StockData.Low))
                            .Append(mlContext.Transforms.CopyColumns("Label", nameof(StockData.Close)))
                            .Append(mlContext.Regression.Trainers.FastTree());
            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var model = pipeline.Fit(trainTestData.TrainSet);
            var predictions = model.Transform(trainTestData.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"R-Squared: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            var predictionResults = mlContext.Data.CreateEnumerable<StockPrediction>(predictions, reuseRowObject: false);
            var testData = mlContext.Data.CreateEnumerable<StockData>(trainTestData.TestSet, reuseRowObject: false);
            foreach(var (predicted, actual) in predictionResults.Zip(testData, (p,a) => (p,a))) {
                Console.WriteLine($"Actual Close Price: {actual.Close}, Predicted Close Price: {predicted.PredictedClose}");
            }
        }
    }
}
