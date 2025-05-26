function [d_prime] = calculate_d_prime(genuine_scores,imposter_scores)   

addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    




%if isempty(counterD)
%        counterD = 1;
%    else
%        counterD = counterD + 1;
%end

    meangen = mean(genuine_scores);
    meanimp = mean(imposter_scores);
    sigmagen = std(genuine_scores);
    sigmaimp = std(imposter_scores);
    d_prime = (sqrt(2) * abs(meangen - meanimp)) / (sqrt((sigmagen^2 + sigmaimp^2)));

end