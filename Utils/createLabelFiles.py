import os.path

import commonTools
import customizedYaml
import pandas as pd
import xml.etree.ElementTree as ET

'''
This script is aiming at tagging the grid files using the cleaned record file.
Some problems we may face and handle in the program:
    1. Some photos are not taken but being recorded.
    2. Some photos are taken but not being recorded.
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


def continue_tagging(record_file, image_dir):
    return 0


if __name__ == '__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_dir = yaml_data.data['base_path']
    grid_dir = yaml_data.build_new_path('base_path', 'grid_images')
    all_data = pd.read_csv(os.path.join(base_dir, 'all_records.csv'), header=None)