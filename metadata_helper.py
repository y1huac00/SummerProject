# Extract metadata from images. Metadata includes image size, class count etc.

import numpy as np
from PIL import Image
import os
import csv
import pandas as pd


def read_image_size(img):
    return img.size


def read_image_file(dir_im):
    return Image.open(dir_im).convert('RGB')


def load_csv(in_path):
    rows = []
    img_species = {}
    img_genuses = {}
    with open(in_path, 'r', encoding='ascii', errors='ignore') as f_in:
        csv_reader = csv.reader(f_in, delimiter=',')
        row1 = next(csv_reader)
        for row in csv_reader:
            row_copy = row.copy()
            img_class = row[1]
            if img_class not in img_species:
                img_species[img_class] = 0
            img_species[img_class] += 1
            image_genus = row_copy[1].split('_')[0]
            if image_genus not in img_genuses:
                img_genuses[image_genus] = 0
            img_genuses[image_genus] += 1
            rows.append(row)
    return rows, row1, img_species, img_genuses


def get_all_size(in_path, out_path, data_path):
    img_classes = {}
    with open(in_path, 'r', encoding='ascii', errors='ignore') as f_in, open(out_path, 'w', encoding='ascii',
                                                                             errors='ignore') as f_out:
        csv_reader = csv.reader(f_in, delimiter=',')
        writer = csv.writer(f_out)
        row1 = next(csv_reader)

        row1.append('width')
        row1.append('height')

        writer.writerow(row1)

        for row in csv_reader:
            row_copy = row.copy()
            img_dir = os.path.join(data_path, row[0])
            img_class = row[1]
            if img_class not in img_classes:
                img_classes[img_class] = 0
            img_classes[img_class] += 1
            img = read_image_file(img_dir)
            width, height = read_image_size(img)
            row_copy.append(width)
            row_copy.append(height)
            writer.writerow(row_copy)
    f_in.close()
    f_out.close()
    return img_classes


def label_genus(in_path, out_path, guide_path, treshold=20):
    genus_collection = {}
    base_count = 0
    rows, row1, img_species, img_genuses = load_csv(in_path)
    with open(out_path, 'w', encoding='ascii', errors='ignore') as f_out:
        writer = csv.writer(f_out)
        for row in rows:
            row_copy = row.copy()
            genus = row_copy[1].split('_')[0]
            if img_genuses[genus] <= treshold:
                continue
            if genus not in genus_collection:
                genus_collection[genus] = base_count
                base_count += 1
            row_copy[1] = genus_collection[genus]
            writer.writerow(row_copy)
    f_out.close()
    with open(guide_path, 'w', encoding='ascii', errors='ignore') as f_guide:
        writer = csv.writer(f_guide)
        for k, v in genus_collection.items():
            row = []
            row.append(k)
            row.append(v)
            writer.writerow(row)
    f_guide.close()


def label_species(in_path, out_path, guide_path, treshold=20):
    species_collection = {}
    base_count = 0
    rows, row1, img_species, img_genuses = load_csv(in_path)
    with open(out_path, 'w', encoding='ascii', errors='ignore') as f_out:
        writer = csv.writer(f_out)
        for row in rows:
            row_copy = row.copy()
            species = row_copy[1]
            if img_species[species] <= treshold:
                continue
            if species not in species_collection:
                species_collection[species] = base_count
                base_count += 1
            row_copy[1] = species_collection[species]
            writer.writerow(row_copy)
    f_out.close()
    with open(guide_path, 'w', encoding='ascii', errors='ignore') as f_guide:
        writer = csv.writer(f_guide)
        for k, v in species_collection.items():
            row = []
            row.append(k)
            row.append(v)
            writer.writerow(row)
    f_guide.close()



def image_by_source():
    '''
    Generate metadata file organized by the source of the image. YPM images would be stored in YPM.csv
    NHMUK files would be stored in NHMUK.csv
    This would help better seprate images from different source to do segmentation
    :return:
    '''
    root = 'Data/'
    opt = 'Data_organized/'

    for folder_name in os.listdir(root):
        if not folder_name.startswith('.'):
            data_folder = os.path.join(os.path.join(root, folder_name), 'data')
            data_file = os.path.join(data_folder,'data.csv')
            df = pd.read_csv(data_file)
            df['scientificName'] = folder_name
            YPM_ids = df.loc[df['institutionCode'] == 'YPM']
            NHM_ids = df.loc[df['institutionCode'] != 'YPM']



            YPM_species = YPM_ids[['file_name','scientificName']]
            YPM_genus = YPM_ids[['file_name','genus']]

            NHM_species = NHM_ids[['file_name', 'scientificName']]
            NHM_genus = NHM_ids[['file_name', 'genus']]

            YPM_species.to_csv(opt+'YPM_species_csv', mode='a', header=False, index=False)
            YPM_genus.to_csv(opt+'YPM_genus_csv', mode='a', header=False, index=False)

            NHM_species.to_csv(opt + 'NHM_species_csv', mode='a', header=False, index=False)
            NHM_genus.to_csv(opt + 'NHM_genus_csv', mode='a', header=False, index=False)


    return 0


if __name__ == '__main__':
    flag = 1
    if flag ==0:
        treshold = 20
        classes = load_csv('input.csv', 'output.csv', './Plaindata')
        print(classes)
        label_genus('input.csv', 'genus.csv', 'genus_guide.csv')
        label_species('input.csv', 'species.csv', 'species_guide.csv')
    image_by_source()
