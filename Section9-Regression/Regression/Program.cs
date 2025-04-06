using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePricePrediction {
    public class HouseData {
        [LoadColumn(0)]
        public float HouseSizeSqft { get; set; }

        [LoadColumn(1)]
        public float NumBedrooms { get; set; }

        [LoadColumn(2)]
        public float NumBathrooms { get; set; }

        [LoadColumn(3)]
        public string Neighborhood { get; set; }

        [LoadColumn(4)]
        public float SalePrice { get; set; }
    }

    public class HousePricePrediction {
        [ColumnName("Score")]
        public float PredictedSalePrice { get; set; }
    }

    class Program {
        static void Main(string[] args) {
            var mlContext = new MLContext(seed: 0);
            var dataPath = Path.Combine(Environment.CurrentDirectory, "house-price-data.csv");
            IDataView dataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, separatorChar: ',', hasHeader: true);
            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;
            var pipeline = mlContext.Transforms.Concatenate("Features", "HouseSizeSqft", "NumBedrooms", "NumBathrooms")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"))
                .Append(mlContext.Transforms.Concatenate("Features", "Features", "Neighborhood"))
                .Append(mlContext.Transforms.CopyColumns("Label", "SalePrice"))
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Label"));
            var trainedModel = pipeline.Fit(trainData);
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(trainedModel);
            var sampleHouseData = new HouseData() {
                HouseSizeSqft = 2000,
                NumBedrooms = 3,
                NumBathrooms = 2,
                Neighborhood = "Southwest"
            };
            var prediction = predictionEngine.Predict(sampleHouseData);
            Console.WriteLine($"Predicted Sale Price: ${prediction.PredictedSalePrice}");
        }
    }
}
