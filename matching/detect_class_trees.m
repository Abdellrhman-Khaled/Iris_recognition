function class = detect_class_trees(img)

addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    


load('tree_model.mat', 'mdl');
csvFilePath = "D:\year3\biometrics\Final_project\feature_extraction.csv";
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
 [treeLabels, treeScores]= predict(mdl, [mean_val, variance_val, energy_val, entropy_val]);
if(treeScores < 70)
    class = 0;
 else
    class = treeLabels;
end

end