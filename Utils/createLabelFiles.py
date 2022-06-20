import os.path

import commonTools
import customizedYaml
import pandas as pd
import xml.etree.ElementTree as ET

import regex as re

'''
This script is aiming at tagging the grid files using the cleaned record file.
Some problems we may face and handle in the program:
    1. Some photos are not taken but being recorded.
    2. Some photos are taken but not being recorded.

Creation process of the files are expected to have following properties:
    1. Returned file should be undre PASCAL VOC format.
    2. Count of the actual ostracods vs the recorded amount should be recorded.
    3. Each grid contains only ONE species. If more than one found, send for error processing.
    4. Species = genus + ' ' + species
'''


def replace_xml_name(new_name, original_xml):
    """
    Replace the object names in the xml file to the tagged names in the PASCAL_VOC annotation file
    """
    objects = original_xml.findall('object')
    obj_cnt = 0
    for obj in objects:
        obj_name = obj.find('name')
        obj_name.text = new_name
        obj_cnt += 1
    return original_xml, obj_cnt

def get_target(target):
    if bool(re.search('genus',target.lower())):
        return 8
    elif bool(re.search('species',target.lower())):
        return 9
    return 0

def build_files(record_file, target):
    """
    This function is building the iconic parts of the files for finding the file names.
    pattern: HKUV12_FD_1_[\d,\dd]X_grid_1.tif

    run first, then efficiency
    if the grid is not recorded:
        if annoation files exists:
            replace('Ostracods', 'unidentified')
            if it is not empty:
                record error
        else:
            create empty annotation file
    if the grid is recorded but without annotation:
        record error

    """
    pattern_dict = {}
    target_index = get_target(target)
    if target_index == 0:
        raise ('Invalid target, should be genus or species')
    for idx, row in record_file.iterrows():
        folder_id_name = row[0] + '_' + row[1] + '_' + '[\d,\dd]X'
        grid_id_name = folder_id_name + '_grid_' + str(row[2])
        data_tuple = (grid_id_name, row[target_index])
        pattern_dict[folder_id_name] = data_tuple
    return pattern_dict


if __name__ == '__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_dir = yaml_data.data['base_path']
    grid_dir = yaml_data.build_new_path('base_path', 'grid_images')
    yaml_data.data['genus_annotation'] = yaml_data.build_new_path('base_bath', 'genus_annotation')
    yaml_data.data['pseudo_annotation'] = yaml_data.build_new_path('base_path', 'pseudo_annotation')
    pseudo_pascal_dir = yaml_data.build_new_path('pseudo_annotation', 'pascal_voc')
    genus_pascal_dir = yaml_data.build_new_path('genus_annotation', 'pascal_voc')
    all_data = pd.read_csv(os.path.join(base_dir, 'all_records.csv'), header=None)
