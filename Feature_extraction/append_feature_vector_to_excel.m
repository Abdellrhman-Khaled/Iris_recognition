function append_feature_vector_to_excel(feature_vector, image_number, output_filename)
        addpath('D:\year3\biometrics\Project_final\Feature_extraction\');
    addpath('D:\year3\biometrics\Project_final\matching\');
    addpath('D:\year3\biometrics\Project_final\models\');
    addpath('D:\year3\biometrics\Project_final\normalized_photos(optional)\');
    addpath('D:\year3\biometrics\Project_final\performance\');
    addpath('D:\year3\biometrics\Project_final\pre-proccessing(optional)\');
    addpath('D:\year3\biometrics\Project_final\test\');
    addpath('D:\year3\biometrics\Project_final\train\');
    

        csv_string = sprintf('%.4f, %d', feature_vector, image_number);

    try
        existing_data = xlsread(output_filename);
        existing_data(end+1, :) = [feature_vector, image_number];
        xlswrite(output_filename, existing_data, 'Sheet1');
        
    catch
        xlswrite(output_filename, [feature_vector, image_number], 'Sheet1', 'A1');
        
    end
end    