using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TimeSeries;

public class Program {

    public class TrafficData {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float Traffic { get; set; }
    }

    public class Prediction {
        public float[] PredictedTraffic { get; set; }
        public float[] LowerBoundTraffic { get; set; }
        public float[] UpperBoundTraffic { get; set; }
    }

    static void EvaluateMetrics(IDataView testData, IDataView predictions, MLContext mlContext) {
        IEnumerable<float> actual = mlContext.Data.CreateEnumerable<TrafficData>(testData, true)
            .Select(observed => observed.Traffic);
        IEnumerable<float> forecast = mlContext.Data.CreateEnumerable<Prediction>(predictions, true)
            .Select(prediction => prediction.PredictedTraffic[0]);
        
        var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);
        var MAE = metrics.Average(Math.Abs);
        var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2)));

        Console.WriteLine("Evaluation Metrics");
        Console.WriteLine("---------------------");
        Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
        Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}");
    }

    static void Forecast(MLContext mlContext, IDataView testData, int horizon, TimeSeriesPredictionEngine<TrafficData, Prediction> forecaster) {
        Prediction forecast = forecaster.Predict(horizon: horizon);
        var testTrafficData = mlContext.Data.CreateEnumerable<TrafficData>(testData, reuseRowObject: false).ToList();
        for(int i = 0; i < horizon && i < forecast.PredictedTraffic.Length; i++) {
            string date = testTrafficData[i].Date.ToShortDateString();
            float actualTraffic = i < testTrafficData.Count ? testTrafficData[i].Traffic : 0;
            float lowerEstimate = Math.Max(0, forecast.LowerBoundTraffic[i]);
            float estimate = forecast.PredictedTraffic[i];
            float upperEstimate = forecast.UpperBoundTraffic[i];
            Console.WriteLine($"Date: {date}\n" + 
                              $"Actual Traffic: {actualTraffic}\n" + 
                              $"Lower Estimate: {lowerEstimate}\n" + 
                              $"Forecast: {estimate}\n" + 
                              $"Upper Estimate: {upperEstimate}\n");
        }
    }

    public static void Main(string[] args) {
        string dataPath = "data.csv";
        var mlContext = new MLContext();
        var dataView = mlContext.Data.LoadFromTextFile<TrafficData>(dataPath, separatorChar: ',', hasHeader: true);
        var filledDataView = mlContext.Transforms.ReplaceMissingValues(
            outputColumnName: nameof(TrafficData.Traffic),
            replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
            .Fit(dataView).Transform(dataView);
        var trainTestData = mlContext.Data.TrainTestSplit(filledDataView, testFraction: 0.2);
        var trainData = trainTestData.TrainSet;
        var testData = trainTestData.TestSet;
        var pipeline = mlContext.Forecasting.ForecastBySsa(
            outputColumnName: nameof(Prediction.PredictedTraffic),
            inputColumnName: nameof(TrafficData.Traffic),
            windowSize: 14,
            seriesLength: 100,
            trainSize: 80,
            horizon: 7,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: nameof(Prediction.LowerBoundTraffic),
            confidenceUpperBoundColumn: nameof(Prediction.UpperBoundTraffic)
        );
        var model = pipeline.Fit(trainData);
        var predictions = model.Transform(testData);
        var forecastingEngine = model.CreateTimeSeriesEngine<TrafficData, Prediction>(mlContext);
        Forecast(mlContext, testData, 7, forecastingEngine);
    }
}
