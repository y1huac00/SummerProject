import cv2
import numpy as np

img  = cv2.imread("D:/pythonproject/ostracod/test/HK14THL1C_104_105_50X.tif")
# img  = cv2.imread("D:/pythonproject/ostracod/test/HK14THL1C_136_137_50X.tif")
# cv2.imshow("Image", img)
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat



scale_percent = 10
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

rotated = rotate_image(resized, 5)
cv2.imshow("rotated", rotated)

gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)


# cv2.imshow("gray", gray)

# image_center = tuple(np.array(gray.shape[1::-1]) / 2)
# rot_mat = cv2.getRotationMatrix2D(image_center, 2, 1.0)
# rotated = cv2.warpAffine(gray, rot_mat, gray.shape[1::-1], flags=cv2.INTER_LINEAR)


blur = cv2.GaussianBlur(gray, (9,9), 0)
cv2.imshow("blur", blur)

blur = cv2.medianBlur(blur,9)
cv2.imshow('median',blur)

blur = cv2.GaussianBlur(blur, (9,9), 0)
cv2.imshow("blur", blur)

blur = cv2.medianBlur(blur,9)
cv2.imshow('median',blur)

# thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)


contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
c = 0
for i in contours:
        area = cv2.contourArea(i)
        print(area)
        if 20000 < area < 30000:
            # if area > max_area:
            print(i)
            max_area = area
            best_cnt = i
            # (x,y,w,h) = cv2.boundingRect(i)
            # cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 2)

            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(rotated, [box], 0, (0, 255, 0), 2)

            # image = cv2.drawContours(resized, contours, c, (0, 255, 0), 2)

        c+=1
print(max_area)
# for contour in contours:
#     (x,y,w,h) = cv2.boundingRect(contour)
#     cv2.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), 2)

# mask = np.zeros((gray.shape),np.uint8)
# cv2.drawContours(mask,[best_cnt],0,255,-1)
# cv2.drawContours(mask,[best_cnt],0,0,2)
# cv2.imshow("mask", mask)
#
# out = np.zeros_like(gray)
# out[mask == 255] = gray[mask == 255]
# cv2.imshow("New image", out)
#
# blur = cv2.GaussianBlur(out, (5,5), 0)
# cv2.imshow("blur1", blur)
#
# thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
# cv2.imshow("thresh1", thresh)
#
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# c = 0
# for i in contours:
#         area = cv2.contourArea(i)
#         if area > 1000/2:
#             cv2.drawContours(resized, contours, c, (0, 255, 0), 2)
#         c+=1


cv2.imshow("Final Image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# dst = np.float32(thresh)
#
# dst = cv2.cornerHarris(dst,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# dst[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',dst)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
