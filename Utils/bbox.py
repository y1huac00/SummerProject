import cv2
import numpy as np
import os
import xml.etree.cElementTree as ET
import tqdm


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def preprocess(image, threshold_lower, threshold_upper, horizontal_kernel, horizontal_kernel_size, square_kernel_size,
               iteration):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.inRange(image, threshold_lower, threshold_upper)
    #cv2.imshow('threshold', image)
    # image = cv2.medianBlur(image,9)
    # ret, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow('blur', image)

    if horizontal_kernel is True:
        kernel = np.ones((1, horizontal_kernel_size), np.uint8)
        kernel_t = np.transpose(kernel)
        for i in range(0, 2):
            image = cv2.erode(image, kernel, iterations=iteration)
            image = cv2.erode(image, kernel_t, iterations=iteration)
            image = cv2.dilate(image, kernel, iterations=iteration + 1)
            image = cv2.dilate(image, kernel_t, iterations=iteration + 1)
    else:
        kernel = np.ones((square_kernel_size, square_kernel_size), np.uint8)
        image = cv2.erode(image, kernel, iterations=iteration)
        image = cv2.dilate(image, kernel, iterations=iteration)

    #cv2.imshow('erode and dilate', image)

    return image


def findcontours(original_image, preprocessed, DRAW, typ):
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursimg = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    arealist = []
    shap = original_image.shape
    bndboxes = []
    rang = (10000, 30000) if typ == 'A' else (2000, 15000)

    for c, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if rang[0] < area < rang[1]:
            arealist.append(area)
            if DRAW is True:
                cv2.drawContours(contoursimg, contours, c, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(contour)
                if (x + w > shap[1] * 0.63 and y + h > shap[0] * 0.73) or x + w > shap[0] - shap[
                    0] * 0.05 or x + w < shap[0] * 0.05 or y + h > shap[1] - shap[1] * 0.05 or y + h < shap[
                    1] * 0.05:
                    continue
                cv2.rectangle(original_image, (x - 10, y - 10), (x + w, y + h), (0, 0, 255), 2)
                bndboxes.append({'xmin': x - 10, 'ymin': y - 10, 'xmax': x + w, 'ymax': y + h})

    #cv2.imshow('contour', contoursimg)
    #cv2.imshow('rectangle box', original_image)
    if DRAW is True:
        return bndboxes
    return contours, len(arealist), len(contours)


def findbestsetting(image, params):
    record = []
    typ = params['type']
    for iteration in params['iteration']:
        for horizontal_kernel_size in params['horizontal_kernel_size']:
            for square_kernel_size in params['square_kernel_size']:
                for threshold_upper in params['threshold_upper']:
                    for threshold_lower in params['threshold_lower']:
                        for horizontal_kernel in params['horizontal_kernel']:
                            preprocesed = preprocess(image, threshold_lower, threshold_upper, horizontal_kernel,
                                                     horizontal_kernel_size, square_kernel_size, iteration)
                            contours, n_box, n_contours = findcontours(image, preprocesed, DRAW=False, typ=typ)
                            # print(
                            #     f'i {iteration}-hkz {horizontal_kernel_size}-sqz {square_kernel_size}-tu {threshold_upper}-tl {threshold_lower}-hk {horizontal_kernel}-t {typ}')
                            # print(f'n_box: {n_box}, n_contours:{n_contours}')
                            # print('---------------------------------------------')
                            if n_contours > 1000:
                                continue
                            record.append({'n_box': n_box, 'iteration': iteration,
                                           'horizontal_kernel_size': horizontal_kernel_size,
                                           'square_kernel_size': square_kernel_size, 'threshold_upper': threshold_upper,
                                           'threshold_lower': threshold_lower, 'horizontal_kernel': horizontal_kernel,
                                           'n_contours': n_contours, 'type': typ})

    # record = dict(sorted(record.items(), key=lambda item: item[1][1]))
    # record = sorted(record, key=lambda d: [-d['n_contours'], d['n_box']], reverse=True)
    # record = sorted(record, key=lambda d: 0.5*(d['n_contours']-d['n_box'])+0.5*d['n_box'])
    record = sorted(record, key=lambda d: 8 * d['n_box'] + 1.2 * d['threshold_lower'] - 1 * d['n_contours'],
                    reverse=True)
    best = record[0]
    # print('best:', best)
    preprocesed = preprocess(image, best['threshold_lower'], best['threshold_upper'], best['horizontal_kernel'],
                             best['horizontal_kernel_size'], best['square_kernel_size'], best['iteration'])

    bndboxes = findcontours(image, preprocesed, DRAW=True, typ=typ)

    return bndboxes


def save2voc(image_folder, file, image, bboxes):
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder').text = os.path.basename(image_folder)
    filename = ET.SubElement(annotation, 'filename').text = file
    path = ET.SubElement(annotation, 'path').text = os.path.join(image_folder, file)
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width').text = f'{image.shape[0]}'
    height = ET.SubElement(size, 'height').text = f'{image.shape[1]}'
    depth = ET.SubElement(size, 'depth').text = f'{image.shape[2]}'
    segmented = ET.SubElement(annotation, 'segmented').text = '0'

    for bbox in bboxes:
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name').text = 'Ostracods'
        pose = ET.SubElement(obj, 'pose').text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated').text = '0'
        difficult = ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        xmin = ET.SubElement(bndbox, 'xmin').text = f'{xmin}'
        ymin = ET.SubElement(bndbox, 'ymin').text = f'{ymin}'
        xmax = ET.SubElement(bndbox, 'xmax').text = f'{xmax}'
        ymax = ET.SubElement(bndbox, 'ymax').text = f'{ymax}'

    tree = ET.ElementTree(annotation)

    tree.write(os.path.join(image_folder, file[:-4] + '.xml'))

    return 0


def semantic_segment(image_folder, file, params, DRAW=False):
    image = cv2.imread(os.path.join(image_folder, file))
    # cv2.imshow('original_image', image)
    # print(image.shape)
    if image.shape[0] > 1250:
        params['type'] = 'A'
    else:
        params['type'] = 'B'
        params['threshold_lower'] = [50, 40, 30]
        # params['threshold_upper'] = [220]

    bndboxes = findbestsetting(image, params)
    # print(bndboxes)

    save2voc(image_folder, file, image, bndboxes)

    if DRAW is True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    all_image_folders = 'E:/HKU_Study/PhD/Lab_work/Keyence_Images'
    # image_folder = '/Users/chenyihua/desktop/pythonprojects/ostracod data/testdata/A/HK14THL1C_64_65_50X/'
    # file = 'HK14THL2C_0_1_50X_grid_2.tif'
    params = {'threshold_lower': [85, 80, 70, 60],
              'threshold_upper': [245],
              'horizontal_kernel': [True],
              'horizontal_kernel_size': [3, 6],
              'square_kernel_size': [3],
              'iteration': [2]
              }

    # Single image testing

    # image_folder = '/Users/chenyihua/desktop/pythonprojects/ostracod data/testdata/A/HK14THL1C_80_81_50X/'
    # file = 'HK14THL1C_80_81_50X_grid_39.tif'
    # semantic_segment(image_folder, file, params, True)

    # mass production

    all_image_folders = '/Users/chenyihua/ostracoddata/testdata/DB2C'
    REPLACE = True  # if replace existing xml files

    # directorylist = [i for i in os.listdir(all_image_folders) if os.path.isdir(os.path.join(all_image_folders, i))]

    directorylist = ['HK14DB2C_96_97_50X']

    for directory in tqdm.tqdm((directorylist)):
        image_folder = os.path.join(all_image_folders, directory)
        for file in files(image_folder):
            if file[-4:] != '.tif':
                continue
            if REPLACE is False:
                if os.path.exists(os.path.join(image_folder, file[:-4]+'.xml')):
                    continue
            index = file.find('grid_')
            num_grid = int(file[index+5:-4])
            if num_grid >= 40:
                continue
            semantic_segment(image_folder, file, params)
        print(directory)

