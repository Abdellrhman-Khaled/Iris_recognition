function deep_model(folderpath)


    addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    


% Define constants
IMAGE_HEIGHT = 480;
IMAGE_WIDTH = 480;
NUM_CLASSES = 35;
EPOCHS = 20;
BATCH_SIZE = 32;
LEARNING_RATE = 0.001;

% Define data directory
DATASET_DIR = 'E:\uni\bio_project\all iris\all iris';

% Load images and labels
classNames = {'iris1', 'iris2', 'iris3', 'iris4', 'iris5','iris6','iris7','iris8','iris9','iris10','iris11','iris12','iris13','iris14','iris15','iris16','iris17','iris18','iris19','iris20','iris21','iris22','iris23','iris24','iris25','iris26','iris27','iris28','iris29','iris30','iris31','iris32','iris33','iris34','iris35'};

features = [];
labels = [];
for i = 1:numel(classNames)
    class_name = classNames{i};
    files_list = dir(fullfile(DATASET_DIR, class_name, '*.bmp'));
    for j = 1:numel(files_list)
        img_path = fullfile(DATASET_DIR, class_name, files_list(j).name);
        img = imread(img_path);
        img = imresize(img, [IMAGE_HEIGHT, IMAGE_WIDTH]);
        img_rgb = repmat(img, [1, 1, 3]); % replicate single channel across three channels
        features = cat(4, features, img_rgb);
        labels = [labels; i];
    end
end
% Convert labels to categorical
labels = categorical(labels);% one hot encoding

% Split data into training and testing sets
cv = cvpartition(labels, 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

% Create training and testing datastores
trainFeatures = features(:,:,:,trainIdx);
trainLabels = labels(trainIdx);
testFeatures = features(:,:,:,testIdx);
testLabels = labels(testIdx);

% Data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10,10], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augmentedTrainImages = augmentedImageDatastore([IMAGE_HEIGHT, IMAGE_WIDTH], trainFeatures, trainLabels, ...
    'DataAugmentation', imageAugmenter);
 % split data into train and validation with validation size 0.15 

 % make another conv layer with 128 and make stride with 1  
% Define CNN architecture
layers = [
    imageInputLayer([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 1)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(NUM_CLASSES)
    softmaxLayer
    classificationLayer
];

% Define training options
options = trainingOptions('adam', ...
    'InitialLearnRate', LEARNING_RATE, ...
    'MaxEpochs', EPOCHS, ...
    'MiniBatchSize', BATCH_SIZE, ...
    'Shuffle', 'every-epoch', ...%
    'Verbose', true, ...
    'Plots', 'training-progress' ...
);

p = parcluster('local');
parpool(p);

% Train the model
trainedNet = trainNetwork(augmentedImages, layers, options);

% Close the parallel pool
delete(gcp('nocreate'))
% Save the trained model
save('beep_model_2.mat', 'trainedNet');
% Evaluate the model on the test set
testImages = augmentedImageDatastore([IMAGE_HEIGHT, IMAGE_WIDTH], testFeatures, testLabels);
testPredictions = classify(trainedNet, testImages);
accuracy = sum(testPredictions == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy)]);

end