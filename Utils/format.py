import os
import shutil
import yaml
import argparse
from tqdm import tqdm


def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            exit(200)


def pseudo_annotation(base_path, dst_path):
    """
    xml files --> dst_path/Pseudo_annotation
    :param base_path: path of all folders
    :param dst_path: destination path
    :return: None
    """
    dst_path = os.path.join(dst_path, 'Pseudo_annotation')
    # get a list of folders
    folders = [i for i in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, i))]
    copied_folders = [i for i in os.listdir(dst_path) if os.path.isdir(os.path.join(dst_path, i))]
    folders = list(set(folders) - set(copied_folders))
    for folder in tqdm(folders):
        # for each folder, get a list of xml files
        xml_files = [i for i in os.listdir(os.path.join(base_path, folder)) if i[-4:] == '.xml']
        txt_files = [i for i in os.listdir(os.path.join(base_path, folder)) if i[-4:] == '.txt']
        xmldst = os.path.join(dst_path, 'pascal_voc', folder)
        yolodst = os.path.join(dst_path, 'yolo', folder)
        os.makedirs(xmldst, exist_ok=True)
        os.makedirs(yolodst, exist_ok=True)
        for xml_file in xml_files:
            # copy each xml file to dst/Pseudo_annotation/pascal_voc
            shutil.copyfile(os.path.join(base_path, folder, xml_file), os.path.join(xmldst, xml_file))

        for txt_file in txt_files:
            # copy each txt file to dst/Pseudo_annotation/yolo
            shutil.copyfile(os.path.join(base_path, folder, txt_file), os.path.join(yolodst, txt_file))

    print('Pseudo_annotation done!')


def grid_images(base_path, dst_path):
    """
    grid images --> dst_path/Grid_images
    :param base_path: path of all folders
    :param dst_path: destination path
    :return: None
    """
    dst_path = os.path.join(dst_path, 'Grid_images')
    # get a list of folders
    folders = [i for i in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, i))]
    copied_folders = [i for i in os.listdir(dst_path) if os.path.isdir(os.path.join(dst_path, i))]
    folders = list(set(folders) - set(copied_folders))
    for folder in tqdm(folders):
        # for each folder, get a list of tif files
        grid_files = [i for i in os.listdir(os.path.join(base_path, folder)) if i[-4:] == '.tif']
        dst = os.path.join(dst_path, folder)
        os.makedirs(dst, exist_ok=True)
        for grid_file in grid_files:
            # save each tif file to dst
            shutil.copyfile(os.path.join(base_path, folder, grid_file), os.path.join(dst, grid_file))
    print('Grid_images done!')


def raw_images(base_path, dst_path):
    """
    raw images --> dst_path/Raw_images
    :param base_path: path of raw images
    :param dst_path: destination path
    :return: None
    """
    dst_path = os.path.join(dst_path, 'Raw_images')
    # get a list of raw images
    images = [i for i in os.listdir(base_path) if i[-4:] in ('.tif', '.jpg')]
    copied_images = [i for i in os.listdir(dst_path) if i[-4:] in ('.tif', '.jpg')]
    images = list(set(images) - set(copied_images))
    for image in tqdm(images):
        # copy each raw image to dst/Raw_images
        shutil.copyfile(os.path.join(base_path, image), os.path.join(dst_path, image))
    print('Raw_images done!')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./format.yaml', help='format yaml path')
    opt = parser.parse_args()
    return opt


def create_folders(dst):
    os.makedirs(os.path.join(dst, 'Pseudo_annotation'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'Species_annotation'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'Raw_images'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'Grid_images'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'Genus_annotation'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'Raw_annotation'), exist_ok=True)


if __name__ == '__main__':
    opt = parse_opt()
    data = read_yaml(opt.yaml)
    create_folders(data['dst_path'])
    raw_images(base_path=data['base_path'], dst_path=data['dst_path'])
    pseudo_annotation(base_path=data['base_path'], dst_path=data['dst_path'])
    grid_images(base_path=data['base_path'], dst_path=data['dst_path'])
    # TODO: Species_annotation
    # TODO: Genus_annotation
    # TODO: Raw_annotation
    print('done!')
