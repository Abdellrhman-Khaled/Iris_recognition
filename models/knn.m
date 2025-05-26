
  function [accuracy_knn] = knn()
    % Specify the path to your CSV file


addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');



    csvFilePath = "D:\year3\biometrics\Project_final\feature_extraction.csv";

    
    % Read the data from the CSV file
    data = xlsread(csvFilePath);

    % Shuffle the data randomly
    rng('default'); % For reproducibility
    shuffledData = data(randperm(size(data, 1)), :);

    % Calculate the split indices
    numSamples = size(shuffledData, 1);
    trainSize = round(0.80 * numSamples); % 80% for training
    testSize = numSamples - trainSize;   % Remaining for testing

    % Split the data
    trainData = shuffledData(1:trainSize, :);
    testData = shuffledData(trainSize+1:end, :);

    % Extract features (X) and labels (Y)
    X_train = trainData(:, 1:4);
    Y_train = trainData(:, 5);

    % Define the hyperparameter grid for KNN
    knnNumNeighbors = [3, 5, 7, 9, 11, 15, 21, 25, 31, 35, 41, 45, 51, 61];
    distanceMetrics = {'euclidean', 'cityblock', 'minkowski', 'chebychev', 'cosine'};
    distanceWeights = {'equal', 'inverse', 'squaredinverse'};

    bestAccuracy = 0;
    bestKNNModel = [];
    
    % Iterate through all combinations of hyperparameters
    for k = 1:length(knnNumNeighbors)
        for d = 1:length(distanceMetrics)
            for w = 1:length(distanceWeights)
                knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', knnNumNeighbors(k), ...
                                   'Distance', distanceMetrics{d}, ...
                                   'DistanceWeight', distanceWeights{w});
                knnPredictions = predict(knnModel, testData(:, 1:4));
                accuracy = sum(knnPredictions == testData(:, 5)) / testSize;
                if accuracy > bestAccuracy
                    bestAccuracy = accuracy;
                    bestKNNModel = knnModel;
                end
            end
        end
    end

    % Save the best KNN model
    save('KNN_model.mat', 'bestKNNModel');

    % Output KNN accuracy
    disp(['KNN Accuracy: ', num2str(bestAccuracy)]);

    % Save the accuracy
    save('accuracyKNN', "bestAccuracy");
    accuracy_knn = bestAccuracy;
end