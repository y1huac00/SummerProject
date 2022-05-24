"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2
import numpy as np
import os
import functools


def read_and_down(file_path, down_factor=32):
    '''
    For read and down scale the images.
    :param file_path: the path for original images
    :param down_factor: the scale down factor for original image
    :return: src: original image; dwn: scaled image
    '''
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    width = int(src.shape[1] // down_factor)
    height = int(src.shape[0] // down_factor)
    dim = (width, height)
    dwn = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
    # dwn = cv2.GaussianBlur(dwn, (3,3), 0)
    return src, dwn


def axis_candidate(projection):
    # Assumed separation: 12*5
    threshold = projection.max() * 0.5
    axis_candidates = []
    for p in range(0, projection.size):
        if projection[p] <= threshold:
            axis_candidates.append(p)
    return axis_candidates


def clean_candidates(candidates, projections, if_x=1):
    axis_breaks = []
    if if_x == 1:
        axis_factor = 12
    else:
        axis_factor = 5
    break_length = projections.size / axis_factor
    for c in range(1, len(candidates)):
        if candidates[c] - candidates[c - 1] > break_length / 3:
            axis_breaks.append(candidates[c - 1])
            axis_breaks.append(candidates[c])
    return axis_breaks


def grid_crop(x_cand, y_cand, trgt, file_string, scale):
    m = len(x_cand)
    n = len(y_cand)
    sub_images = []
    print(m * n)
    for y in range(0, n - 1, 2):
        for x in range(0, m - 1, 2):
            sub_images.append(trgt[y_cand[y] * scale:y_cand[y + 1] * scale, x_cand[x] * scale:x_cand[x + 1] * scale])
    p = 1
    for img in sub_images:
        fs = file_string + str(p) + '.tif'
        cv2.imwrite(fs, img)
        p += 1


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def sep_image(img_file, img_folder, thr_value=160, scale=32):
    img_root = ''.join(img_file.split(".")[:-1])
    img, scaled = read_and_down(os.path.join(img_folder, img_file), scale)

    thr = cv2.threshold(scaled, thr_value, 255, cv2.THRESH_BINARY_INV)[1]
    h_proj = thr.sum(axis=1)
    w_proj = thr.sum(axis=0)
    cnd_x = clean_candidates(axis_candidate(w_proj), w_proj, 1)
    cnd_y = clean_candidates(axis_candidate(h_proj), h_proj, 0)

    root_folder = os.path.join(img_folder, img_root)
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)
    # else:
    #     # already cropped
    #     return
    file_string = os.path.join(root_folder, (img_root + '_grid_'))

    grid_crop(cnd_x, cnd_y, img, file_string, scale)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def resize(img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def preprocess(img, type, pblurmedian, pthreshold, pdilate):
    if type == 'A':
        resized = resize(img, 10)
        # resized = rotate_image(resized, 2)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (pblurmedian, pblurmedian), 0)
        cv2.imshow("blur", blur)

        blur = cv2.medianBlur(blur, pblurmedian)
        cv2.imshow('median', blur)

        blur = cv2.GaussianBlur(blur, (pblurmedian, pblurmedian), 0)
        cv2.imshow("blur", blur)

        blur = cv2.medianBlur(blur, pblurmedian)
        cv2.imshow('median', blur)

        ret, thresh = cv2.threshold(blur, pthreshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)

        kernel = np.ones((1, pdilate), np.uint8)  # note this is a horizontal kernel
        kernel = np.transpose(kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        kernel = np.ones((1, pdilate), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imshow("dilated", thresh)

        return resized, thresh

    elif type == 'B':
        resized = resize(img, 20)
        # resized = rotate_image(resized, 4)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (pblurmedian, pblurmedian), 0)
        cv2.imshow("blur", blur)

        blur = cv2.medianBlur(blur, pblurmedian)
        cv2.imshow('median', blur)

        # blur = cv2.GaussianBlur(blur, (9, 9), 0)
        # cv2.imshow("blur", blur)
        #
        # blur = cv2.medianBlur(blur, 9)
        # cv2.imshow('median', blur)

        ret, thresh = cv2.threshold(blur, pthreshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)

        kernel = np.ones((1, pdilate), np.uint8)  # note this is a horizontal kernel
        kernel = np.transpose(kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        kernel = np.ones((1, pdilate), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        # thresh = cv2.erode(thresh, kernel, iterations=1)
        cv2.imshow("dilated", thresh)

        return resized, thresh


def findcontours(draw_img, preprocessed_img, lower, upper):
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arealist = []
    rectarealist = []
    rectlist = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if lower < area < upper:
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # check if contour is square-like or something else
            xdiff = max(abs(box[0, 0] - box[1, 0]), abs(box[0, 0] - box[2, 0]), abs(box[0, 0] - box[3, 0]))
            ydiff = max(abs(box[0, 1] - box[1, 1]), abs(box[0, 1] - box[2, 1]), abs(box[0, 1] - box[3, 1]))
            if ydiff * 0.5 < xdiff < ydiff * 1.5:
                arealist.append(area)
                rectarealist.append(cv2.contourArea(box))
                rectlist.append(rect)

    arealist = sorted(arealist)
    rectarealist = sorted(rectarealist)

    # print('arealist:', arealist)
    print('grids detected:', len(arealist))
    # print('median:', arealist[round(len(arealist)/2)])

    # print('rectarealist:', rectarealist)
    # print('median:', rectarealist[round(len(rectarealist)/2)])

    # cv2.imshow("Contour", contour_img)

    return (contours, rectarealist), rectlist


def drawcontour(contours, draw_img, contour_img, lower, upper):
    c = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if lower < area < upper:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            xdiff = max(abs(box[0, 0] - box[1, 0]), abs(box[0, 0] - box[2, 0]), abs(box[0, 0] - box[3, 0]))
            ydiff = max(abs(box[0, 1] - box[1, 1]), abs(box[0, 1] - box[2, 1]), abs(box[0, 1] - box[3, 1]))
            if ydiff * 0.5 < xdiff < ydiff * 1.5:
                cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)
                cv2.drawContours(contour_img, contours, c, (0, 0, 255), 2)
        c += 1
    cv2.imshow("Contour", contour_img)


def evaluate(contourlist, paramslist):
    # print([np.var(np.array(i[1])) for i in contourlist if len(i[1]) == 60])
    if 60 not in [len(i[1]) for i in contourlist]:
        return False, 0
    minvar = min([np.var(np.array(i[1]) / np.linalg.norm(np.array(i[1]))) for i in contourlist if len(i[1]) == 60])
    # print(minvar)
    c = 0
    for contour in contourlist:
        if len(contour[1]) == 60 and np.var(np.array(contour[1]) / np.linalg.norm(
                np.array(contour[1]))) == minvar:  # if grid detected is 60 and smallest var
            print(
                f'Best params: blurmedian: {paramslist[c][0]}, threshold: {paramslist[c][1]}, dilate: {paramslist[c][2]}, with normalized variance {minvar}')
            return contour[0], c
        c += 1
    return False, c


def findbestcontours(img, rang, params, type):
    best_contours = None
    # find contours in different params
    for blurmedian in params[type]['blurmedian']:
        for threshold in params[type]['threshold']:
            contourslist = []
            contour_imglist = []
            resized = None
            paramslist = []
            rectlistlist = []

            for dilate in params[type]['dilate']:
                print(f'--- blurmedian: {blurmedian}, threshold: {threshold}, dilate: {dilate} ---')
                resized, preprocessed_img = preprocess(img, type, blurmedian, threshold, dilate)
                contour_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
                contour_imglist.append(contour_img)
                contourscandidate, rectlist = findcontours(resized, preprocessed_img, lower=rang[0], upper=rang[1])
                contourslist.append(contourscandidate)
                paramslist.append([blurmedian, threshold, dilate])
                rectlistlist.append(rectlist)

            best_contours, c = evaluate(contourslist, paramslist)
            if best_contours is not False:
                return best_contours, resized, contour_imglist[c], rectlistlist[c]

    return None, None, None, None


def straighten(img, rectlist):
    imgheight, imgwidth = img.shape[:2]  # image shape has 3 dimensions
    image_center = (
        imgwidth / 2,
        imgheight / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    anglelist = np.array([90 - i[2] if i[2] > 45 else i[2] for i in rectlist])
    angle = 0 - np.average(anglelist)
    print(anglelist, angle)

    M = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    cv2.imshow('rotated', resize(rotated_image, 10))

    return rotated_image, angle

def crop(rotated_image, best_rectlist, type):
    scale = 10 if type == 'A' else 5
    def sort(a, b):
        if 0.8 * b[0][1] < a[0][1] < 1.2 * b[0][1]:
            if a[0][0] < b[0][0]:
                return -1
            else:
                return 1
        else:
            return -1
    best_rectlist = sorted(best_rectlist, key=functools.cmp_to_key(sort))
    for index, rect in enumerate(best_rectlist):
        center = (int(rect[0][0] * scale), int(rect[0][1] * scale))
        width = int((rect[1][0]+3) * scale)
        height = int((rect[1][1]+3) * scale)

        cropped = cv2.getRectSubPix(
            rotated_image, (height, width), center)

        fs = f'D:/pythonproject/ostracod/test/testt/{index+1}.tif'
        cv2.imwrite(fs, cropped)



def solutionB(file, type):  # single file for test
    img = cv2.imread(file)
    cv2.imshow('original image', resize(img, 10 if type == 'A' else 20))
    rang = (13000, 30000) if type == 'A' else (22000, 40000)

    params = {'A': {'blurmedian': [5, 7, 9], 'threshold': [150, 160, 170, 180], 'dilate': [3, 5, 6, 7, 8, 10]},
              'B': {'blurmedian': [3, 5], 'threshold': [150, 160], 'dilate': [3, 5, 6, 7, 8]}
              }

    # find best contours from different params (Current criteria: grids == 60 and minimum variance of rectangle area)
    best_contours, resized, contour_img, best_rectlist = findbestcontours(img, rang, params, type)
    if best_contours is None:
        print('No 60 detected')
        return 6
    # draw best contours on the resized image
    drawcontour(best_contours, draw_img=resized, contour_img=contour_img, lower=rang[0], upper=rang[1])

    rotated_image, angle = straighten(img, best_rectlist)

    if abs(angle) < 0.3:
        crop(img, best_rectlist, type)
    else:
        best_contours, resized, contour_img, best_rectlist = findbestcontours(rotated_image, rang, params, type)
        if best_contours is None:
            print('No 60 detected')
            return 6
        # draw best contours on the resized image
        drawcontour(best_contours, draw_img=resized, contour_img=contour_img, lower=rang[0], upper=rang[1])
        crop(rotated_image, best_rectlist, type)

    cv2.imshow("Final Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO: modify SolutionA to work with straightened image
    # TODO: SolutionB to every file in a given directory


if __name__ == '__main__':
    A = False
    if A:  # Solution A: select candidates
        img_folder = 'E:/HKU_Study/PhD/Lab_work/Keyence_Images'
        for file in files(img_folder):
            print(file)
            sep_image(file, img_folder, 160, 16)
    else:  # Solution B: grid contour
        sampleA = ('D:/pythonproject/ostracod/test/HK14DB1C_136_137_50X.tif', 'A')
        sampleB = ('D:/pythonproject/ostracod/test/HK14THL1C_136_137_50X.tif', 'B')
        solutionB(sampleA[0], sampleA[1])
