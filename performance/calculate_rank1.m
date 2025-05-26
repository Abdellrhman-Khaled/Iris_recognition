function [rank1rate] = calculate_rank1(genuine_scores)

addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    

 

    genuine_scores = genuine_scores(:);
    flooredgen = floor(genuine_scores);
    threshold = mode(flooredgen);
    highscores = sum(flooredgen >= threshold);
    GenLength = length(genuine_scores);
    rank1rate = ( highscores / GenLength );
 
end

