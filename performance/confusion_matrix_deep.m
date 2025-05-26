function [confusion_matrix] = calculate_confusion_matrix_deep()


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
load('beep_model.mat', 'trainedNet');
dataset_folder = 'D:\year3\biometrics\Project_final\test\';
classNames = {'iris1', 'iris2', 'iris3', 'iris4', 'iris5','iris6','iris7','iris8','iris9','iris10','iris11','iris12','iris13','iris14','iris15','iris16','iris17','iris18','iris19','iris20','iris21','iris22','iris23','iris24','iris25','iris26','iris27','iris28','iris29','iris30','iris31','iris32','iris33','iris34','iris35'};

confusion_matrix = zeros(35, 140);


for i = 1:size(confusion_matrix, 2) 
    x=1;
    for j = 1:size(confusion_matrix, 1) 
         if(x>35)
             x=1;
         end
        image_path = fullfile(dataset_folder, [num2str(i) '.png']);
        img = imread(image_path);
        newImage = imresize(img, [IMAGE_HEIGHT, IMAGE_WIDTH]); % Resize the image
        img_rgb = repmat(newImage, [1, 1, 3]);
        predictedProbs = predict(trainedNet, img_rgb);
        predictedLabel = classify(trainedNet, img_rgb);
        confusion_matrix(i, j) = predictedProbs(x);
        x=x+1;
    end

end

confusion_matrix = confusion_matrix';
end

