import pandas as pd
import commonTools
import customizedYaml
import os
import shutil

'''
The file is created for getting the files that require attention for re-tagging. An error csv file is provided for
reference. Out put of this script should be a folder holding files pending for fixing.

This script should also has the ability to:
    1. Record the fixed and un-fixed files, pack the files pending for repairing.
    2. Move the fixed files back to the dataset and replace the ones containing error
    3. update the fixed files in the error recording.

Some requirements for the design:
    a. Separate I/O and CPU work into different functions.
    b. Avoid using static variable. All static variables should be read from yaml file.
'''


def build_folder_name(core, slide):
    return core + '_' + slide


def move_to_error_bucket(source, dest):
    if not os.path.exists(dest):
        os.mkdir(dest)
    shutil.copy2(source, dest)

def getting_error(error_out_path, grid_path, error_info):
    # Possible magnify for files: 50X, 40X, 80X, 100X
    if 'processed' not in error_info:
        error_info['processed'] = 0
    for idx, row in error_info.iterrows():
        core = row['core']
        slide = row['slide']
        grid = row['grid']
        root_name = build_folder_name(core, slide)
        source_folder_name = build_folder_name(root_name, '50X')
        # Find $base_dir/grid_images/HK14TLH1C_0_1_50X/HK14TLH1C_0_1_50X_grid_1.tif
        # for folder in commonTools.conditional_folders(grid_path, target_folder_name):
        #     # Folder should be: HK14TLH1C_0_1_50X
        #     grid_file_path = os.path.join(grid_path, folder)
        #     magnify = grid_file_path.split('_')[-1]
        #     source_path = grid_file_path
        source_grid = root_name +'_50X'+ '_grid_' + str(grid) + '.tif'
        # $base_dir/errors/HK14TLH1C_0_1_50X
        dest_folder_path = os.path.join(error_out_path, source_folder_name)
        # $base_dir/grid_images/HK14TLH1C_0_1_50X
        source_folder_path = os.path.join(grid_path, source_folder_name)
        # $base_dir / grid_images / HK14TLH1C_0_1_50X / HK14TLH1C_0_1_50X_grid_1.tif
        final_source_path = os.path.join(source_folder_path, source_grid)
        if os.path.isfile(final_source_path):
            move_to_error_bucket(final_source_path, dest_folder_path)
            error_info.loc[idx,'processed'] = 1
    return error_info


if __name__ == '__main__':
    '''
    All IO should be handled in main or specially designed IO functions
    '''
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_path = yaml_data.get_data('base_path')
    error_path = yaml_data.get_data('error_path')
    grid_path = yaml_data.build_new_path('base_path', 'grid_images')
    error_out_path = yaml_data.build_new_path('base_path', 'errors')
    error_info = pd.read_csv(error_path)
    error_info = getting_error(error_out_path, grid_path, error_info)
    error_info.to_csv(error_path, index=False)
