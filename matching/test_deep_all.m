function [classname,Probs] = test_deep_all(img)


addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');





IMAGE_HEIGHT = 480;
IMAGE_WIDTH = 480;
% Load the trained model
load('beep_model_2.mat', 'trainedNet');
classNames = {'iris1', 'iris2', 'iris3', 'iris4', 'iris5','iris6','iris7','iris8','iris9','iris10','iris11','iris12','iris13','iris14','iris15','iris16','iris17','iris18','iris19','iris20','iris21','iris22','iris23','iris24','iris25','iris26','iris27','iris28','iris29','iris30','iris31','iris32','iris33','iris34','iris35'};
% Read and preprocess the new image
newImage = imread(img);
newImage = imresize(newImage, [IMAGE_HEIGHT, IMAGE_WIDTH]); % Resize the image


if size(newImage,3)==3
    img_rgb=newImage;
else
 img_rgb = repmat(newImage, [1, 1, 3]);
end




predictedProbs = predict(trainedNet, img_rgb);
predictedLabel = classify(trainedNet, img_rgb);
% Display the probabilities for each class
classname = classNames{predictedLabel};
Probs = predictedProbs(predictedLabel);

if(Probs < 0.80)
Probs=0;
classname="Imposter";
end


disp(classname);
disp(Probs)
end