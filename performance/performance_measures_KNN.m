function [acc1,genuine_scores,imposter_scores,GenImpImage,tpr,fpr,ROCImage,eer_KNN,eerImage,d_prime_KNN,tmr1_KNN,tmr001_KNN,rank1rate_KNN] = performance_measures_KNN()


addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    

load('accuracyKNN.mat',"bestAccuracy");
acc1 = bestAccuracy;

eer_KNN = 0;
d_prime_KNN = 0;
tmr1_KNN = 0;
tmr001_KNN = 0;
rank1rate_KNN = 0;

load ('transposed.mat','a');

% in order to handle rank 1 identification flooring.
a = a*100;

[genuine_scores, imposter_scores] = get_gen_imp_vectors(a);

[GenImpImage] = GenImp(genuine_scores,imposter_scores);

[tpr,fpr,ROCImage] = GenImpROC(genuine_scores,imposter_scores);

[eer,eerImage] = compute_eer(tpr,fpr);

[d_prime] = calculate_d_prime(genuine_scores,imposter_scores);

[tmr1,tmr001] = calculate_tmr_fmr(tpr,fpr);

[rank1rate] = calculate_rank1(genuine_scores);

eer_KNN = eer;
d_prime_KNN = d_prime;
rank1rate_KNN = rank1rate;
tmr1_KNN = tmr1;
tmr001_KNN = tmr001;


end


