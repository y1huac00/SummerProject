import os

import customizedYaml
import commonTools

'''
Replace all THL with TLH.
'''


def correct_file_name(old_name, wrong, correct):
    new_name = old_name.replace(wrong, correct)
    return new_name


def process_grid_names(grid_path, wrong='THL', correct='TLH'):
    for folders in commonTools.conditional_folders(grid_path, wrong):
        new_folder_name = correct_file_name(folders, wrong, correct)
        full_folder_path = os.path.join(grid_path, folders)
        for files in commonTools.files(full_folder_path):
            new_file_name = correct_file_name(files, wrong, correct)
            os.rename(os.path.join(full_folder_path, files), os.path.join(full_folder_path, new_file_name))
        new_folder_path = os.path.join(grid_path, new_folder_name)
        os.rename(full_folder_path, new_folder_path)


def process_image_names(image_path, wrong='THL', correct='TLH'):
    for files in commonTools.conditional_files(image_path, wrong):
        new_file_name = correct_file_name(files, wrong, correct)
        new_file_path = os.path.join(image_path, new_file_name)
        old_file_path = os.path.join(image_path, files)
        os.rename(old_file_path,new_file_path)

def process_pseudo_annotations(annotation_path, wrong, correct):
    for folders in commonTools.conditional_folders(annotation_path, wrong):
        new_folder_name = correct_file_name(folders, wrong, correct)
        full_folder_path = os.path.join(annotation_path, folders)
        for files in commonTools.files(full_folder_path):
            new_file_name = correct_file_name(files, wrong, correct)
            os.rename(os.path.join(full_folder_path, files), os.path.join(full_folder_path, new_file_name))
        new_folder_path = os.path.join(annotation_path, new_folder_name)
        os.rename(full_folder_path, new_folder_path)

if __name__ == '__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_dir = yaml_data.data['base_path']
    grid_path = yaml_data.build_new_path('base_path', 'grid_images')
    image_path = yaml_data.build_new_path('base_path','raw_images')
    wrong = 'FD'
    correct = 'FD_'
    process_grid_names(grid_path, wrong, correct)
    process_image_names(image_path, wrong, correct)
