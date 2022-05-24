import cv2
import numpy as np


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

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
        resized = rotate_image(resized, 2)

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
        resized = rotate_image(resized, 4)

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

    arealist = sorted(arealist)
    rectarealist = sorted(rectarealist)

    # print('arealist:', arealist)
    print('grids detected:', len(arealist))
    # print('median:', arealist[round(len(arealist)/2)])

    # print('rectarealist:', rectarealist)
    # print('median:', rectarealist[round(len(rectarealist)/2)])

    # cv2.imshow("Contour", contour_img)

    return (contours, rectarealist)


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
    minvar = min([np.var(np.array(i[1])) for i in contourlist if len(i[1]) == 60])
    # print(minvar)
    c = 0
    for contour in contourlist:
        if len(contour[1]) == 60 and np.var(np.array(contour[1])) == minvar:  # if grid detected is 60 and smallest var
            print(f'Best params: blurmedian: {paramslist[c][0]}, threshold: {paramslist[c][1]}, dilate: {paramslist[c][2]}, with variance {minvar}')
            return contour[0], c
        c += 1
    return False, c


def findbestcontours(img, rang, params, type):
    contourslist = []
    best_contours = None
    # find contours in different params
    for blurmedian in params[type]['blurmedian']:
        for threshold in params[type]['threshold']:
            contour_imglist = []
            resized = None
            paramslist = []

            for dilate in params[type]['dilate']:
                print(f'--- blurmedian: {blurmedian}, threshold: {threshold}, dilate: {dilate} ---')
                resized, preprocessed_img = preprocess(img, type, blurmedian, threshold, dilate)
                contour_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
                contour_imglist.append(contour_img)
                contourslist.append(findcontours(resized, preprocessed_img, lower=rang[0], upper=rang[1]))
                paramslist.append([blurmedian, threshold, dilate])

            best_contours, c = evaluate(contourslist, paramslist)
            if best_contours is not False:
                return best_contours, resized, contour_imglist[c]

    return None, None, None


def solutionB(file, type):  # single file for test
    img = cv2.imread(file)
    rang = (13000, 30000) if type == 'A' else (22000, 40000)

    params = {'A': {'blurmedian': [5, 7, 9], 'threshold': [150, 160, 170, 180], 'dilate': [3, 5, 6, 7, 8]},
              'B': {'blurmedian': [3, 5], 'threshold': [150, 160], 'dilate': [5, 6, 7, 8]}
              }

    # find best contours from different params (Current criteria: grids == 60 and minimum variance of rectangle area)
    best_contours, resized, contour_img = findbestcontours(img, rang, params, type)
    if best_contours is None:
        print('No 60 detected')
        return 6
    # draw best contours on the resized image
    drawcontour(best_contours, draw_img=resized, contour_img=contour_img, lower=rang[0], upper=rang[1])

    cv2.imshow("Final Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO: straighten every grid and crop


if __name__ == '__main__':
    sampleA = ('D:/pythonproject/ostracod/test/HK14PCR1C_56_57_50X.tif', 'A')
    sampleB = ('D:/pythonproject/ostracod/test/HK14THL1C_136_137_50X.tif', 'B')

    solutionB(sampleB[0], sampleB[1])
