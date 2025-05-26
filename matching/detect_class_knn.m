function class = detect_class_knn(img)
addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    

load('KNN_model.mat', 'bestKNNModel');
csvFilePath = "D:\year3\biometrics\Project_final\feature_extraction.csv";
data = xlsread(csvFilePath);
%img = imread(image_path);

if size(img,3)==3
    img=rgb2gray(img);
end


maxWidth = 500;
[height, width, ~] = size(img);
if width > maxWidth
    scaleFactor = maxWidth / width;
    img = imresize(img, scaleFactor);
end

normalized_image=Normalization_Daugman(img);
[mean_val, variance_val, energy_val, entropy_val] = Gabor_features_extractor1(normalized_image);
 [knnPredictedLabels, knnScores]= predict(bestKNNModel, [mean_val, variance_val, energy_val, entropy_val]);
 if(knnScores < 0.70)
    class = 0;
 else
    class = knnPredictedLabels;
 end
% display(knnScores);
end