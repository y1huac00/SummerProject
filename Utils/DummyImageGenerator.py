from PIL import Image
import math
import os
import csv
from PIL import ImageDraw

'''
Description: This script is collection of methods related to pasting interested objects into selected backgrounds.
'''


def MinMax(min, max):
    '''
    Return larger/smaller value
    Trick: put variables with mini/max into corresponding place
    :param min: item submitted for compare
    :param max: item submitted for compare
    :return: Mini or Max value
    '''
    if min > max:
        return min
    return max


def BKGGen(w_max, h_max, mp, ind=0):
    '''
    Creating background images for pasting training objects, could be extended by adding more case
    :param w_max: max width of images in input image batch
    :param h_max: max height of images in input image batch
    :param mp: multiplier of back ground image, should be square root of input image batch size
    :param ind: indicator of choice of background generation function
    :return: background image generated
    '''

    ''' Python 3.10 syntax
    match ind:
        case '0':
            return Image.new('RGB', (mp * w_max, mp * h_max), (0, 0, 0))
        case '1':
            return Image.new('RGB', (mp * w_max, mp * h_max), (255, 255, 255))
        case _:
            return Image.new('RGB', (mp * w_max, mp * h_max), (0, 0, 0))
    '''
    if ind == 0:
        return Image.new('RGB', (mp * w_max, mp * h_max), (0, 0, 0))
    elif ind == 1:
        return Image.new('RGB', (mp * w_max, mp * h_max), (255, 0, 255))

    return Image.new('RGB', (mp * w_max, mp * h_max), (0, 0, 0))


def PasteFunc(imgs, bkg, ind=1):
    """
    Extend here for more pasting methods
    :param imgs: Array of images for pasting
    :param bkg: Background for images
    :param ind: indicator for pasting algorithm
    :return: Pasted image with interested objects
    """
    if ind == 0:
        return 0
    return 0


def ROICaculation(w, h, x, y, w_b, h_b):
    """
    Create scaled Region Of Interest for yolo context
    Each row is class x_center y_center width height format.
    :param w: width of object image
    :param h: height of object image
    :param x: width of pasted origin point
    :param y: height of pasted origin point
    :param w_b: width of the background image
    :param h_b: height of the background image
    :return: [x_center, y_center, width, height]
    """
    roi_x = (0.5 * w + x) / w_b
    roi_y = (0.5 * h + y) / h_b
    roi_w = w / w_b
    roi_h = h / h_b
    return roi_x, roi_y, roi_w, roi_h


def GenerateDummy(images, bkgi=1, loc=0, pred=[]):
    '''
    :param loc: Locating indicator for location functions
    :param bkg: background generating function
    :param images: List of path to images
    :return: a dummy images contained image for training, bkg is the dummy image and roi is the region of interest
    '''
    # Possible optimization: create fixed grid to contain images to reduce memory consumption
    img_array = []  # Using queue would be better
    roi = []
    w_max = 0
    h_max = 0
    cnt = 0
    mv = len(images)
    mp = math.ceil(math.sqrt(mv))
    for img in images:
        i = Image.open(img)
        w, h = i.size
        w_max = MinMax(w, w_max)
        h_max = MinMax(h, h_max)
        img_array.append(i)
    # Place to plugin background providing functions
    bkg = BKGGen(w_max, h_max, mp, bkgi)
    w_b, h_b = bkg.size
    for row in range(0, mp):
        if cnt <= mv - 1:
            for col in range(0, mp):
                wi, hi = img_array[cnt].size
                centerX = int((-wi + (2 * row + 1) * w_max) / 2)
                centerY = int((-hi + (2 * col + 1) * h_max) / 2)
                # LIFO: Possibile bug if labeled in FIFO order
                bkg.paste(img_array[cnt], (centerX, centerY))
                # Debug only: Print image path on the image to find bug
                # ImageDraw.Draw(bkg).text((centerX, centerY), images[cnt], (255, 255, 255))
                # ImageDraw.Draw(bkg).text((centerX, centerY), pred[cnt], (255, 255, 255))
                roi.append((ROICaculation(wi, hi, centerX, centerY, w_b, h_b)))
                cnt += 1
                if cnt >= mv:
                    break
        else:
            break
    return bkg, roi


def TestFunc():
    ''' Just test'''
    image_path = ('../Plaindata/')
    ref_path = ('../Results/1625212258species_result.csv')
    imgs = []
    tags_pred = []
    with open(ref_path, 'r', encoding='ascii', errors='ignore') as f_in:
        csv_reader = csv.reader(f_in, delimiter=',')
        for i in range(0, 14):
            row = next(csv_reader)
            #imgs.append(os.path.join(image_path, row[0]))
            imgs.append('.'+row[0])
            tags_pred.append(row[1])
    f_in.close()
    res, _ = GenerateDummy(imgs,0,0,tags_pred)
    res.save('result.jpg', quality=100)

TestFunc()