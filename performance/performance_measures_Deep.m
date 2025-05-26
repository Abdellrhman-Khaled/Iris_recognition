function [acc1,genuine_scores,imposter_scores,GenImpImage,tpr,fpr,ROCImage,eer_Deep,eerImage,d_prime_Deep,tmr1_Deep,tmr001_Deep,rank1rate_Deep] = performance_measures_Deep()

    addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    

acc1 = 0.9688;


eer_Deep = 0;
d_prime_Deep = 0;
tmr1_Deep = 0;
tmr001_Deep = 0;
rank1rate_Deep = 0;

load ('deepMatrix.mat');

[genuine_scores, imposter_scores] = get_gen_imp_vectors(con);

[GenImpImage] = GenImp(genuine_scores,imposter_scores);

[tpr,fpr,ROCImage] = GenImpROC(genuine_scores,imposter_scores);

[eer,eerImage] = compute_eer(tpr,fpr);

[d_prime] = calculate_d_prime(genuine_scores,imposter_scores);

[tmr1,tmr001] = calculate_tmr_fmr(tpr,fpr);

[rank1rate] = calculate_rank1(genuine_scores);

eer_Deep = eer;
d_prime_Deep = d_prime;
rank1rate_Deep = rank1rate;
tmr1_Deep = tmr1;
tmr001_Deep = tmr001;

end
