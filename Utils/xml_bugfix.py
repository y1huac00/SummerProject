import xml.etree.cElementTree as ET
import os

base_path = '/Users/chenyihua/ostracoddata/testdata/DB2C'
all_folders = [i for i in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, i))]

print(all_folders)

for folder in all_folders:
    xml_folder = os.path.join(base_path, folder)
    xml_files = [i for i in os.listdir(xml_folder) if i[-4:] == '.xml']
    print(xml_files)

    for xml_file in xml_files:
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        for i in root.findall('path'):
            database = i.find('database')
            if database is None:
                continue
            else:
                database_name = database.text
                root.remove(i)
                image_folder = root.find('folder').text
                filename = root.find('filename').text
                folder_name = os.path.basename(image_folder)
                path_name = os.path.join(image_folder, filename)
                print(folder_name)
                print(filename)
                print(path_name)

                root.find('folder').text = folder_name
                root.find('filename').text = filename
                path = ET.SubElement(root, 'path').text = path_name

                source = ET.SubElement(root, 'source')
                database_ = ET.SubElement(source, 'database').text = database_name

        tree = ET.ElementTree(root)

        tree.write(os.path.join(xml_folder, xml_file))

