function [tpr,fpr,ROCImage] = GenImpROC(gen,imp)
   

addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    



genuine_scores = gen;
imposter_scores = imp;

genuine_scores = genuine_scores(:);
imposter_scores = imposter_scores(:);
 thresholds = unique([genuine_scores; imposter_scores]);
    num_thresholds = length(thresholds);
    tpr = zeros(num_thresholds, 1);
    fpr = zeros(num_thresholds, 1);
    for i = 1:num_thresholds
        tpr(i) = sum(genuine_scores >= thresholds(i)) / length(genuine_scores);
        fpr(i) = sum(imposter_scores >= thresholds(i)) / length(imposter_scores);
    end
    
  

    ROC = figure('Visible','off'); %da el myfigure handle 
    plot(fpr,tpr,LineWidth=2);
    title('Receiver Operating Characteristic (ROC) Curve');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    xlim([0 1]);
    ylim([0 1]);    
    grid on;
   
    frame = getframe(gcf);

    ROCImage = frame.cdata;

    close(ROC);

end
