function [tmr1,tmr001] = calculate_tmr_fmr(tpr,fpr)

addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    




fmr1 = 0.01;
fmr001 = 0.0001;

[~, idx1] = min(abs(fpr - fmr1));
[~, idx2] = min(abs(fpr - fmr001));

tmr1 = tpr(idx1);
tmr001 = tpr(idx2);



end
