import os
import xml.etree.cElementTree as ET
import shutil

base_path = '/Users/chenyihua/ostracoddata/testdata/DB2C'
yolo_label_path = '/Users/chenyihua/ostracoddata/testdata/yolo/testing1/labels'
yolo_image_path = '/Users/chenyihua/ostracoddata/testdata/yolo/testing1/images'

folders = [i for i in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, i))]

for folder in folders:
    xml_files = [i for i in os.listdir(os.path.join(base_path, folder)) if i[-4:] == '.xml']

    for xml_file in xml_files:
        tree = ET.parse(os.path.join(base_path,folder,xml_file))
        root = tree.getroot()

        objects = root.findall('object')

        if len(objects) > 0:
            width = int(root.find('size').findtext('width'))
            height = int(root.find('size').findtext('height'))

            for object in root.findall('object'):
                bndbox = object.find('bndbox')
                xmin = int(bndbox.findtext('xmin'))
                xmax = int(bndbox.findtext('xmax'))
                ymin = int(bndbox.findtext('ymin'))
                ymax = int(bndbox.findtext('ymax'))
                centerx = (xmin + xmax) / 2
                centerx /= width
                centery = (ymin + ymax) / 2
                centery /= height
                objectwidth = xmax - xmin
                objectwidth /= width
                objectheight = ymax - ymin
                objectheight /= height
                with open(os.path.join(yolo_label_path, xml_file[:-4]) + '.txt', 'a') as f:
                    f.write(f'0 {centerx} {centery} {objectwidth} {objectheight}\n')

            # copy image file to yolo path
            shutil.copyfile(os.path.join(base_path, folder, xml_file[:-4]) + '.tif', os.path.join(yolo_image_path, xml_file[:-4]) + '.tif')
