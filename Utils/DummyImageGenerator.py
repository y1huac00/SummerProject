from PIL import Image
import math
import os
import csv
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
    elif ind ==1:
        return Image.new('RGB', (mp * w_max, mp * h_max), (255, 255, 255))

    return Image.new('RGB', (mp * w_max, mp * h_max), (0, 0, 0))


def PasteFunc(imgs, bkg, ind=0):
    '''
    :param imgs: Array of images for pasting
    :param bkg: Background for images
    :param ind: indicator for pasting algorithm
    :return: Pasted image with interested objects
    '''
    if ind == 0:
        return 0
    return 0


def GenerateDummy(images, bkgi=0, loc=0):
    '''
    :param loc: Locating indicator for location functions
    :param bkg: background generating function
    :param images: List of path to images
    :return: a dummy images contained image for training
    '''
    # Possible optimization: create fixed grid to contain images to reduce memory consumption
    img_array = []  # Using queue would be better
    w_max = 0
    h_max = 0
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
    for row in range(0, mp):
        if mv >= 1:
            for col in range(0, mp):
                wi, hi = img_array[mv - 1].size
                centerX = int((-wi + (2*row+1) * w_max) / 2)
                centerY = int((-hi + (2*col+1) * w_max) / 2)
                bkg.paste(img_array[mv - 1], (centerX, centerY))
                mv = mv - 1
                if mv < 1:
                    break
        else:
            break
    return bkg

def TestFunc():
    image_path = ('../Plaindata/')
    ref_path = ('../Species.csv')
    imgs = []
    with open(ref_path, 'r', encoding='ascii', errors='ignore') as f_in:
        csv_reader = csv.reader(f_in, delimiter=',')
        for i in range(0,9):
            row = next(csv_reader)
            imgs.append(os.path.join(image_path, row[0]))
    f_in.close()
    res = GenerateDummy(imgs)
    res.save('result.jpg', quality=100)

TestFunc()

