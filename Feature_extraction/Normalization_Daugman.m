function [out_img] = Normalization_Daugman(iris_img)
addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
       


polar_radius = 165; % Adjust as needed
    angular_resolution = 360; % Adjust as needed

    [rows, cols] = size(iris_img);
    center_x = cols / 2;
    center_y = rows / 2;
    theta = linspace(0, 2 * pi, cols);
    r = linspace(0, polar_radius, rows);

    % Generate polar coordinates grid
    [X, Y] = meshgrid(1:cols, 1:rows);
    X = X - center_x;
    Y = center_y - Y;

    polar_iris = zeros(rows, cols, 'double');
    for i = 1:cols
        % Compute polar coordinates for this angle
        x = center_x + r .* cos(theta(i));
        y = center_y - r .* sin(theta(i));

        % Interpolate values from Cartesian to polar coordinates
        polar_values = interp2(double(iris_img), x, y, 'linear', 0);

        % Assign interpolated values to polar image
        polar_iris(:, i) = polar_values;
    end

    % Normalize polar image
    polar_iris = mat2gray(polar_iris);

    % Convert back to uint8
    out_img = uint8(255 * polar_iris);
end
