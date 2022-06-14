import os
import customizedYaml
import argparse
# import xml.etree.ElementTree as ET

"""
Place this yolobbox.py and yolobbox.yaml into yolov5 directory
"""


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./yolobbox.yaml', help='folder path yaml path')
    opt = parser.parse_args()
    return opt


def detectsingle(file):
    os.system(f'python detect.py --weights testing1_best.pt --source {file} --save-txt')


def detectall(base_path, result_path):
    folders = [i for i in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, i))]  # get a list of folders
    for folder in folders:
        path = os.path.join(base_path, folder)
        print(path)
        files = os.path.join(path, '*.tif')
        print(files)
        # print(f'python detect.py --weights testing1_best.pt --source {path} --save-txt --project {path} --name yololabels')
        # os.system(f'python detect.py --weights testing1_best.pt --source {path} --save-txt --project {path} --name yololabels')
        os.system(f'python detect.py --weights testing1_best.pt --source {path} --save-txt --project {result_path} --name {folder}')


# def convert():
#     """
#     */yololabels/*.txt --> */voclabels/*.xml
#     """
#     annotation = ET.Element('annotation')
#     folder = ET.SubElement(annotation, 'folder').text = os.path.basename(image_folder)
#     filename = ET.SubElement(annotation, 'filename').text = file
#     path = ET.SubElement(annotation, 'path').text = os.path.join(image_folder, file)
#     source = ET.SubElement(annotation, 'source')
#     database = ET.SubElement(source, 'database').text = 'Unknown'
#     size = ET.SubElement(annotation, 'size')
#     width = ET.SubElement(size, 'width').text = f'{image.shape[0]}'
#     height = ET.SubElement(size, 'height').text = f'{image.shape[1]}'
#     depth = ET.SubElement(size, 'depth').text = f'{image.shape[2]}'
#     segmented = ET.SubElement(annotation, 'segmented').text = '0'
#
#     for bbox in bboxes:
#         obj = ET.SubElement(annotation, 'object')
#         name = ET.SubElement(obj, 'name').text = 'Ostracods'
#         pose = ET.SubElement(obj, 'pose').text = 'Unspecified'
#         truncated = ET.SubElement(obj, 'truncated').text = '0'
#         difficult = ET.SubElement(obj, 'difficult').text = '0'
#         bndbox = ET.SubElement(obj, 'bndbox')
#         xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
#         xmin = ET.SubElement(bndbox, 'xmin').text = f'{xmin}'
#         ymin = ET.SubElement(bndbox, 'ymin').text = f'{ymin}'
#         xmax = ET.SubElement(bndbox, 'xmax').text = f'{xmax}'
#         ymax = ET.SubElement(bndbox, 'ymax').text = f'{ymax}'
#
#     tree = ET.ElementTree(annotation)
#
#     tree.write(os.path.join(image_folder, file[:-4] + '.xml'))


def main():
    opt = parse_opt()
    yaml_data = customizedYaml.yaml_handler(opt.yaml)
    data = yaml_data.data
    data['result_path'] = yaml_data.build_new_path('bas_path','Pseudo_annotation/yolo/')
    detectall(data['base_path'], data['result_path'])
    # TODO: convert txt to pascal vox to dst path


if __name__ == '__main__':
    main()
