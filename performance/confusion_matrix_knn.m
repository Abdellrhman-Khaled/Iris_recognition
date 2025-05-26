function [transposed] = calculate_confusion_matrix_knn()



addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    


dataset_folder = 'D:\year3\biometrics\Project_final\train\';
confusion_matrix = zeros(35, 140);

load('KNN_model.mat', 'knnModelBest');

for i = 1:size(confusion_matrix, 2) 
    x=1;
    for j = 1:size(confusion_matrix, 1) 
        if(x>35)
            x=1;
        end

        image_path = fullfile(dataset_folder, [num2str(i) '.png']);
        img = imread(image_path);
        normalized_image = Normalization_Daugman(img);
        [mean_val, variance_val, energy_val, entropy_val] = Gabor_features_extractor1(normalized_image);
        
        
        
        
        [knnPredictedLabels, knnScores]= predict(knnModelBest, [mean_val, variance_val, energy_val, entropy_val]);
        confusion_matrix(i, j) = knnScores(x);
        x = x+1
    end
end

transposed = confusion_matrix';
end


