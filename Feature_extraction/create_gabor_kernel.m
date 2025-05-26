function gabor_kernel = create_gabor_kernel(lambda, theta, sigma)

addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    


% Create a Gabor filter kernel
    % Adjust the kernel size as needed
    kernel_size = 5;  % Example: 5x5 kernel
    [X, Y] = meshgrid(-floor(kernel_size/2):floor(kernel_size/2));
    gabor_kernel = exp(-(X.^2 + Y.^2) / (2 * sigma^2)) .* cos(2 * pi * lambda * X * cos(theta));
end