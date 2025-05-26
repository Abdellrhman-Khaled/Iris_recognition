function [gen,imp] = get_gen_imp_vectors(inputmatrix)


addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    


gen = zeros(1, 140);

k=1;
for i = 1:140
    gen(i)=inputmatrix(k , i);
    if(rem(i, 4) == 0 )
    k =k+1;
    end

end

imp=zeros(1,4760);



for i = 1:35
    ignore_start =(i - 1) * 4 + 1;
    ignore_end = ignore_start+3;
    k= ((i-1)*140) +1;   
    for j=1:140
        if((j>=ignore_start) && (j<=ignore_end)  )
            continue;
        
        else
            imp(k+j)=inputmatrix(i , j);
        end
    end

end

end












