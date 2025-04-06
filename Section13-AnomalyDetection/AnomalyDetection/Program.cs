using Microsoft.ML;
using Microsoft.ML.Data;

namespace NetworkTrafficAnomalyDetection {
    class Program {
        public class NetworkTrafficData {
            [LoadColumn(0)]
            public string Timestamp { get; set; }

            [LoadColumn(1)]
            public string SourceIP { get; set; }

            [LoadColumn(2)]
            public string DestinationIP { get; set; }

            [LoadColumn(3)]
            public string Protocol { get; set; }

            [LoadColumn(4)]
            public float PacketSize { get; set; }

            [LoadColumn(5)]
            public string Label { get; set; }
        }

        public class NetworkTrafficPrediction {
            [ColumnName("PredictedLabel")]
            public uint PredictedClusterId { get; set; }

            public float[] Score { get; set; }
        }

        static void Main(string[] args) {
            var mlContext = new MLContext();
            var dataPath = "network_data.csv";
            var dataView = mlContext.Data.LoadFromTextFile<NetworkTrafficData>(dataPath, separatorChar: ',', hasHeader: true);
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("SourceIP")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("DestinationIP"))
                .Append(mlContext.Transforms.Concatenate("Features", "PacketSize"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));
            var model = pipeline.Fit(dataView);
            var predictions = model.Transform(dataView);
            var predictedData = mlContext.Data.CreateEnumerable<NetworkTrafficPrediction>(predictions, reuseRowObject: false);
            var actualData = mlContext.Data.CreateEnumerable<NetworkTrafficData>(dataView, reuseRowObject: false);
            using(var predictedEnumerator = predictedData.GetEnumerator())
            using(var actualEnumerator = actualData.GetEnumerator()) {
                while(predictedEnumerator.MoveNext() && actualEnumerator.MoveNext()) {
                    var prediction = predictedEnumerator.Current;
                    var actual = actualEnumerator.Current;
                    var predictedLabel = prediction.PredictedClusterId == 1 ? "Normal" : "Anomalous";
                    Console.WriteLine($"Actual Label: {actual.Label}, Predicted Label: {predictedLabel}, Score: {string.Join(", ", prediction.Score)}");
                }
            }
            Console.WriteLine("Anomaly detection complete.");
        }
    }
}
