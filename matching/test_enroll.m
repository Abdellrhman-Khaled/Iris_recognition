function test_enroll(lable,img,class)
addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');





load('beep_model_2.mat','trainedNet');
load('names.mat', 'Names');
folderPath4= 'D:\year3\biometrics\Project_final\enroll_test_all.csv';
IMAGE_HEIGHT = 480;
IMAGE_WIDTH = 480;
%class = {class};
% Read and preprocess the new image
newImage = imread(img);
newImage = imresize(newImage, [IMAGE_HEIGHT, IMAGE_WIDTH]); % Resize the image

% Display the layers of the network
%disp(trainedNet.Layers);

features = activations(trainedNet, newImage,'fc_2');

% Display the size of the extracted features
%for i = 1:35
 %   fprintf('%.2f%%\n', features(i));
%end


features=reshape(features, 1, 35);
disp(size(features));

append_feature_vector_to_excel(features,lable, folderPath4);
flag = 1;
disp(Names);
for i = 1:length(Names)
        if strcmp(Names{i}, class)
            flag = 0;
            break;  % Exit loop if class label already exists
        end
end

if flag==1
    Names=[Names,class];
end

save('names.mat', 'Names');
end
