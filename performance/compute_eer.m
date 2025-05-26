function [eer,eerImage] = compute_eer(tpr, fpr)


addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    




    far = fpr;
    frr = (1-tpr);

    [mindistance,indexofmin] = min(abs(far-frr));

    eer_far = far(indexofmin);
    eer_frr = frr(indexofmin);

    x_line = [0, 1];
    y_line = [0, 1];

    eerfig = figure('Visible','off');
    plot(far,frr,'Color','red','LineWidth',2);
    hold on;
    plot(eer_far, eer_frr, 'bo', 'MarkerSize', 10, 'LineWidth', 2,'Color','green');
    hold on;
    plot(x_line, y_line, 'r--', 'LineWidth', 2,'Color','blue');
    hold on;
    %hold off;
    

xlabel('FAR');
ylabel('FRR');
title('EER');
legend('FAR and FRR', 'FAR = FRR','EER');

    frame = getframe(gcf);

    eerImage = frame.cdata;
    
    close(eerfig);
    
    eer = ((far(indexofmin) + frr(indexofmin)) / 2);
   
end