function [name,prob] = compare_test(img)
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
% Read and preprocess the new image
newImage = imread(img);
newImage = imresize(newImage, [IMAGE_HEIGHT, IMAGE_WIDTH]); % Resize the image

% Display the layers of the network
%disp(trainedNet.Layers);

features = activations(trainedNet, newImage,'fc_2');

features=reshape(features, 1, 35);
data = xlsread(folderPath4);
max_prob=-1;
final_entry=-1;
for i=1:size(data)
    input=data(i,:);
    label=input(end);
    input=input(1:end-1);
    similarity = dot( features , input' ) / (norm(features) * norm(input));
    probability_same = (similarity + 1) / 2;
    
    if probability_same > max_prob
        max_prob=probability_same;
        final_entry=label;
    end
        
end
if max_prob < 0.79
   % disp("enroll");
   name="not";
   prob=0;
else
    disp(Names(final_entry));
    name = Names(final_entry);
    %disp(max_prob);
    prob=max_prob;
end
end
