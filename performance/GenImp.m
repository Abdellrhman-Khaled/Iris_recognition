function [GenImpImage] = GenImp(genuine_scores,imposter_scores)
    
addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    



[f_genuine, x_genuine] = ksdensity(genuine_scores);

[f_imposter, x_imposter] = ksdensity(imposter_scores);

normalizedXimposter = mapminmax(x_imposter);
normalizedXgenuine = mapminmax(x_genuine);
normalizedYimposter = mapminmax (f_imposter);
normalizedYgenuine = mapminmax (f_genuine);

GenImpfig = figure('Visible','off');
plot(normalizedXgenuine, normalizedYgenuine,'LineWidth', 2, 'Color', 'blue');
hold on;
plot(normalizedXimposter, normalizedYimposter, 'LineWidth', 2, 'Color', 'red');


xlabel('Scores');
ylabel('Density');
title('Distribution of Genuine and Imposter Scores');
legend('Genuine Scores', 'Imposter Scores');

ylim([-1 1.1])
xlim([-1.1 1.1]);
xticks(-1:0.2:1)
hold off;


frame = getframe(gcf);

GenImpImage = frame.cdata;

close(GenImpfig);

end
