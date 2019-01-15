namespace TaxiFarePrediction

module Demo = 

    open System
    open System.IO

    open Microsoft.ML
    open Microsoft.ML.Data

    open Microsoft.ML.Core.Data
    open Microsoft.ML.Runtime.Data
    open Microsoft.ML.Transforms.Normalizers
    open Microsoft.ML.Runtime.Api

    [<CLIMutable>]
    type TaxiTrip = {
        VendorId : string
        RateCode : string
        PassengerCount : float32
        TripTime : float32
        TripDistance : float32
        PaymentType : string
        FareAmount : float32
        }

    [<CLIMutable>]
    type TaxiTripFarePrediction = {
        [<ColumnName("Score")>]
        FareAmount : float32
        }

    [<RequireQualifiedAccess>]
    module Pipeline = 

        let append (estimator : IEstimator<'a>) (pipeline : IEstimator<'b>)  = 
            match pipeline with
            | :? IEstimator<ITransformer> as p -> 
                p.Append estimator
            | _ -> failwith "The pipeline has to be an instance of IEstimator<ITransformer>."

    [<RequireQualifiedAccess>]
    module Peek = 

        open System.Reflection

        let vectorColumn
            (columnName, numberOfRows) 
            (mlContext: MLContext, dataView: IDataView, pipeline: IEstimator<_>) = 
        
                printfn "Peek data in DataView: : Show %i rows with just the '%s' column" numberOfRows columnName

                let transformer = pipeline.Fit (dataView)
                let transformedData = transformer.Transform (dataView)

                transformedData.GetColumn<float32[]>(mlContext, columnName)
                |> Seq.truncate numberOfRows
                |> Seq.iter (fun row ->                
                    row 
                    |> Seq.map (fun x -> sprintf "%f" x) 
                    |> String.concat " | " 
                    |> printfn "%s"
                    printfn ""
                    )

        let dataView<'TObservation 
            when 'TObservation : (new : unit -> 'TObservation) 
            and 'TObservation : not struct>
            (numberOfRows: int)
            (mlContext: MLContext, dataView: IDataView , pipeline: IEstimator<_>) =
        
            printfn "Peek data in DataView: Showing %i rows with the columns specified by TObservation class" numberOfRows

            let transformer = pipeline.Fit(dataView)
            let transformedData = transformer.Transform(dataView)

            transformedData.AsEnumerable<'TObservation>(mlContext, reuseRowObject = false)
            |> Seq.truncate numberOfRows
            |> Seq.toArray
            |> Seq.iteri (fun i row ->
                printfn "Row %i " i
                let fieldsInRow = 
                    row.GetType().GetFields(
                        BindingFlags.Instance |||
                        BindingFlags.Static |||
                        BindingFlags.NonPublic |||
                        BindingFlags.Public
                        )
                fieldsInRow
                |> Seq.iter (fun field -> 
                    printfn "  %s: %A" field.Name (field.GetValue(row))
                    )
                )
        
    [<EntryPoint>]
    let main argv =

        let mlContext = MLContext (seed = Nullable 0)

        let textLoader = 
            mlContext.Data.TextReader(
                TextLoader.Arguments(
                    Separator = ",",
                    HasHeader = true,
                    Column = 
                        [|
                            TextLoader.Column("VendorId", Nullable DataKind.Text, 0)
                            TextLoader.Column("RateCode", Nullable DataKind.Text, 1)
                            TextLoader.Column("PassengerCount", Nullable DataKind.R4, 2)
                            TextLoader.Column("TripTime", Nullable DataKind.R4, 3)
                            TextLoader.Column("TripDistance", Nullable DataKind.R4, 4)
                            TextLoader.Column("PaymentType", Nullable DataKind.Text, 5)
                            TextLoader.Column("FareAmount", Nullable DataKind.R4, 6)
                        |]
                    )
                )

        let rootFolder = 
            Environment.GetCommandLineArgs().[0]
            |> Path.GetDirectoryName
            |> fun basePath -> Path.Combine(basePath, @"../../../../")

        let dataFolder = 
            Path.Combine(rootFolder, "Data")

        let modelsFolder = 
            Path.Combine(rootFolder, "Models")

        let trainDataPath = Path.Combine(dataFolder, "taxi-fare-test.csv")
        
        let train = textLoader.Read trainDataPath
        let filtered = mlContext.Data.FilterByColumn(train, "FareAmount", lowerBound = 1.0, upperBound = 150.0)
        
        let dataProcessPipeline =
            mlContext.Transforms.CopyColumns("FareAmount", "Label")
            |> Pipeline.append (mlContext.Transforms.Categorical.OneHotEncoding("VendorId", "VendorIdEncoded"))
            |> Pipeline.append (mlContext.Transforms.Categorical.OneHotEncoding("RateCode", "RateCodeEncoded"))
            |> Pipeline.append (mlContext.Transforms.Categorical.OneHotEncoding("PaymentType", "PaymentTypeEncoded"))
            |> Pipeline.append (mlContext.Transforms.Normalize(inputName = "PassengerCount", mode = NormalizingEstimator.NormalizerMode.MeanVariance))
            |> Pipeline.append (mlContext.Transforms.Normalize(inputName = "TripTime", mode = NormalizingEstimator.NormalizerMode.MeanVariance))
            |> Pipeline.append (mlContext.Transforms.Normalize(inputName = "TripDistance", mode = NormalizingEstimator.NormalizerMode.MeanVariance))
            |> Pipeline.append (mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", "PassengerCount", "TripTime", "TripDistance"))
        
        (mlContext, filtered, dataProcessPipeline)
        |> Peek.dataView<TaxiTrip> 5 
        
        (mlContext, filtered, dataProcessPipeline)
        |> Peek.vectorColumn ("Features", 5)

        let trainer = mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn = "Label", featureColumn = "Features")
        let trainingPipeline = dataProcessPipeline.Append(trainer)
        
        let trainedModel = trainingPipeline.Fit filtered

        let testDataPath = Path.Combine(dataFolder, "taxi-fare-test.csv")
        let test = textLoader.Read testDataPath

        let predictions = trainedModel.Transform test
        let metrics = mlContext.Regression.Evaluate(predictions, label = "Label", score = "Score")

        metrics.LossFn |> printfn "Loss function: %.2f"
        metrics.RSquared |> printfn "R2 score %.2f"
        metrics.L1 |> printfn "Absolute loss %.2f"
        metrics.L2 |> printfn "Squared loss %.2f"
        metrics.Rms |> printfn "RMS loss %.2f"

        let modelPath = Path.Combine(modelsFolder, "taxi-fare-model.zip")
        
        using (modelPath |> File.Create) 
            (fun modelFile -> trainedModel.SaveTo (mlContext, modelFile))

        let rehydratedModel = 
            let stream = new FileStream (modelPath, FileMode.Open, FileAccess.Read, FileShare.Read)
            mlContext.Model.Load stream

        let predictor = rehydratedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext)
        
        let taxiTripSample = 
            {
                VendorId = "VTS"
                RateCode = "1"
                PassengerCount = 1.0f
                TripTime = 1140.0f
                TripDistance = 3.75f
                PaymentType = "CRD"
                FareAmount = 0.0f 
            }

        predictor.Predict taxiTripSample |> printfn "%A"

        printfn "Press any key to finish."
        Console.ReadKey () |> ignore

        0 // return an integer exit code    