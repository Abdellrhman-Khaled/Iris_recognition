



dataset_folder = 'Dataset';
folderPath='D:\year3\biometrics\Final_project\Cropped_iris\';
folderPath2= 'D:\year3\biometrics\Final_project\Normalized_Daugman\';

all_images = cell(1, 30);
normalized_images = cell(1, 30);

for i = 1:30
    image_path = fullfile(dataset_folder, [num2str(i) '.jpg']);
    img = imread(image_path);
    maxWidth = 500;
    [height, width, ~] = size(img);
    if width > maxWidth
        scaleFactor = maxWidth / width;
        img = imresize(img, scaleFactor);
    end
    
    gray_img= rgb2gray(img);
    
    enhanced_img = histeq(gray_img);
    blurred_img = imgaussfilt(enhanced_img, 3); % Adjust the filter size as needed
    
    sharpenedImage = imsharpen(blurred_img);
    sharpenedImage = imsharpen(sharpenedImage);
    
    all_images{i}=segmentation_canny_hough_transform(sharpenedImage ,img);
    normalized_images{i}=Normalization_Daugman(all_images{i});

    output_path1 = sprintf('%siris_%d.png', folderPath, i);
    
    %imwrite( all_images{i}, output_path1);

    output_path2 = sprintf('%sNormalized_%d.png', folderPath2, i);
   % imwrite( normalized_images{i}, output_path2);


end














