
from DummyImageGenerator import GenerateDummy
import random
import os
import csv

def CreateTrays(len, n):
    """
    Create trays for inference
    :param len: length of input array
    :param n: number of trays to be created
    :return: List of n trays
    """
    tray_list = []
    tray_0 = []
    for i in range(0,len):
        tray_0.append(i)
    tray_list.append(tray_0)
    for j in range(1,n):
        random.seed(j)
        tray_x = tray_0.copy()
        random.shuffle(tray_x)
        tray_list.append(tray_x)
    return tray_list

if __name__ == '__main__':
    image_path = '../Plaindata/'
    ref_path = '../Species.csv'
    out_labels = '../YoloImages/Labels/Train/'
    out_images = '../YoloImages/Images/Train/'
    imgs = []
    classes = []
    batch: int = 9
    with open(ref_path, 'r', encoding='ascii', errors='ignore') as f_in:
        csv_reader = csv.reader(f_in, delimiter=',')
        for row in csv_reader:
            imgs.append(os.path.join(image_path, row[0]))
            classes.append(row[1])
    f_in.close()
    ctr = len(classes) #controlling factor for testing
    trays = CreateTrays(ctr, batch)
    for t in range(0, ctr):
        img_batch = []
        class_batch = []
        for q in range(0, batch):
            img_batch.append(imgs[trays[q][t]])
            class_batch.append(classes[trays[q][t]])
        yolo_img, yolo_roi = GenerateDummy(img_batch)
        yolo_img.save(out_images+'yolo_' + str(t) + '.jpg', quality=100)
        with open(out_labels + 'yolo_' + str(t) + '.txt', 'w', encoding='ascii', errors='ignore') as f_out:
            for roi, cls in zip(yolo_roi,class_batch):
                roi_str = str(cls)+' '+' '.join(str(r) for r in roi)
                f_out.write(roi_str)
                f_out.write('\n')
        f_out.close()
