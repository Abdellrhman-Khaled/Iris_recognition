function [mean_val, variance_val, energy_val, entropy_val] = Gabor_features_extractor1(normalized_iris)
addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    



% Parameters for Gabor filter bank
    num_orientations = 10;
    num_scales = 6;
    lambda = 9;
    sigma = 2;

    % Create Gabor filter bank (same as before)
    % ...
     gabor_bank = cell(num_orientations, num_scales);
    for o = 1:num_orientations
        for s = 1:num_scales
            theta = (o - 1) * pi / num_orientations;
            % Create a Gabor filter kernel
            gabor_kernel = create_gabor_kernel(lambda, theta, sigma);
            gabor_bank{o, s} = gabor_kernel;
        end
    end
    % Apply Gabor filters to the normalized iris image (same as before)
    % ...
     filtered_responses = zeros(size(normalized_iris, 1), size(normalized_iris, 2), num_orientations * num_scales);
    for o = 1:num_orientations
        for s = 1:num_scales
            filtered_responses(:, :, (o - 1) * num_scales + s) = ...
                imfilter(normalized_iris, gabor_bank{o, s}, 'symmetric');
        end
    end

    % Compute statistical measures for the entire filtered response
    mean_val = mean(filtered_responses(:));
    variance_val = var(filtered_responses(:));
    energy_val = sum(filtered_responses(:).^2);
    entropy_val = -sum(filtered_responses(:) .* log2(filtered_responses(:) + eps));
end



