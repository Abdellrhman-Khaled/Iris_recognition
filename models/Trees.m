function[accuracy_tree ] =  Trees()


addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    



% Specify the path to your CSV file
csvFilePath = 'D:\year3\biometrics\Project_final\feature_extraction.csv';

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
X = trainData(:, 1:4);
Y = trainData(:, 5);

% Define hyperparameter search space
paramGrid = struct('MaxNumSplits', [2, 5, 10], ...
                   'MinLeafSize', [1, 5, 10]);

% Define cross-validation options
cv = cvpartition(Y, 'KFold', 5);

% Perform hyperparameter optimization using fitcensemble
t = templateTree();
mdl = fitcensemble(X, Y, 'OptimizeHyperparameters', {'MaxNumSplits', 'MinLeafSize'}, ...
                    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'KFold', 5));

% Predict class labels for test data
X_test = testData(:, 1:4);
Y_test = testData(:, 5);
predictedLabels = predict(mdl, X_test);

% Calculate accuracy
correctPredictions = sum(predictedLabels == Y_test);
totalTestSamples = size(testData, 1);
accuracy_tree = correctPredictions / totalTestSamples;
save('tree_model.mat', 'mdl');
% Output accuracy
%disp(['Accuracy: ', num2str(accuracy)]);
save('accuracy_trees',"accuracy_tree");
end