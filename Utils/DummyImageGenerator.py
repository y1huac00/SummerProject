from PIL import Image
import math
import os

image_path = ('./Plaindata/')


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
    match ind:
        case 0:
            return Image.new('RGB', (mp * w_max, mp * h_max), (0, 0, 0))
        case 1:
            return Image.new('RGB', (mp * w_max, mp * h_max), (255, 255, 255))
        case _:
            return Image.new('RGB', (mp * w_max, mp * h_max), (0, 0, 0))


def GenerateDummy(images, bkg=0, loc=0):
    '''
    :param bkg: background generating function
    :param images: List of path to images
    :return: a dummy images contained image for training
    '''
    # Possible optimization: create fixed grid to contain images to reduce memory consumption
    img_array = []
    w_max = 0
    h_max = 0
    mp = math.ceil(math.sqrt(len(images)))
    for img in images:
        i = Image.open(img)
        w, h = i.size
        w_max = MinMax(w, w_max)
        h_max = MinMax(h, h_max)
        img_array.append[i]
    # Place to plugin background providing functions
    bkg = BKGGen(w_max, h_max, mp, 0)
    
