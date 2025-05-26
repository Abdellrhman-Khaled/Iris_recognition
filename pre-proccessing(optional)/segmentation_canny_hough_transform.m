function [out_img] =segmentation_canny_hough_transform(sharpenedImage , img)
    edges = edge(sharpenedImage, 'Canny', 0.2); % Adjust the thresholds as needed
    [centers, radii] = imfindcircles(edges, [30 150], 'ObjectPolarity', 'bright', 'Sensitivity', 0.9);
    figure;
    imshow(img);
    hold on;
    viscircles(centers, radii, 'EdgeColor', 'b');
    title('Detected Iris');



    %----------------cropping--------------


    iris_center = centers(1, :); % Assuming the first detected circle is the iris
    iris_radius = radii(1);
    
    % Define the cropping region
    x1 = iris_center(1) - iris_radius;
    x2 = iris_center(1) + iris_radius;
    y1 = iris_center(2) - iris_radius;
    y2 = iris_center(2) + iris_radius;
    
    % Crop the iris
    cropped_iris = sharpenedImage(y1:y2, x1:x2, :);
    %figure;
    %imshow(cropped_iris);
    %title('cropped Iris');
    
      sharpenedImage = imsharpen(cropped_iris);
     brightenedImage = imlocalbrighten(sharpenedImage);

    % figure;
    %imshow(brightenedImage);
    %title('Brightened cropped Iris');

    out_img =brightenedImage;