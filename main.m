 function main(folderpath , outpath)
    addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    
    dataset_folder = folderpath;
    %'D:\year3\biometrics\Project_final\train\'; 
    folderPath2= 'D:\year3\biometrics\Project_final\normalized_photos(optional)\';
    folderPath4= outpath;
    %'D:\year3\biometrics\Project_final\feature_extraction.csv';
    
    
    normalized_images = cell(1, 181);
    normalized_images_aug = cell(1, 181);
    normalized_images_aug2 = cell(1, 181);
    normalized_images_aug3 = cell(1, 181);
    
    
    augmentation = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandYReflection',true, ...
        'RandXTranslation',[-5 5], ...
        'RandYTranslation',[-5 5]);
    augmentation2 = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandYReflection',true, ...
        'RandXTranslation',[-10 10], ...
        'RandYTranslation',[-10 10]);
    augmentation3 = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandYReflection',true, ...
        'RandXTranslation',[-7 7], ...
        'RandYTranslation',[-7 7]);
    for i = 1:35
        for j=1:15
            image_path = fullfile(dataset_folder, ['iris' num2str(i) '_' num2str(j) '.bmp']);
            img = imread(image_path);
            maxWidth = 500;
            [height, width, ~] = size(img);
            if width > maxWidth
                scaleFactor = maxWidth / width;
                img = imresize(img, scaleFactor);
            end
            
            augmentedImage = augment(augmentation, img);
            augmentedImage2 = augment(augmentation2, img);
            augmentedImage3 = augment(augmentation3, img);
            normalized_images{i}=Normalization_Daugman(img);
            normalized_images_aug{i}=Normalization_Daugman(augmentedImage);
            normalized_images_aug2{i}=Normalization_Daugman(augmentedImage2);
            normalized_images_aug3{i}=Normalization_Daugman(augmentedImage3);
            
              %output_path2 = sprintf('%sNormalized_%d_%d.png', folderPath2, i,j);
             %imwrite( normalized_images{i}, output_path2);
            
            [mean_val, variance_val, energy_val, entropy_val] = Gabor_features_extractor1(normalized_images{i});
            append_feature_vector_to_excel([mean_val, variance_val, energy_val, entropy_val], i, folderPath4);
    
            [mean_val, variance_val, energy_val, entropy_val] = Gabor_features_extractor1(normalized_images_aug{i});
            append_feature_vector_to_excel([mean_val, variance_val, energy_val, entropy_val], i, folderPath4);
          
            [mean_val, variance_val, energy_val, entropy_val] = Gabor_features_extractor1(normalized_images_aug2{i});
            append_feature_vector_to_excel([mean_val, variance_val, energy_val, entropy_val], i, folderPath4);
            
            [mean_val, variance_val, energy_val, entropy_val] = Gabor_features_extractor1(normalized_images_aug3{i});
            append_feature_vector_to_excel([mean_val, variance_val, energy_val, entropy_val], i, folderPath4);
        end
    end

end









